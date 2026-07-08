from __future__ import annotations

"""Homogenization helpers for the CRC pipeline.

This module contains only the logic related to computing or honoring the
homogenized columns:
    - z_flag_homogenized
    - instrument_type_homogenized
    - object_type_homogenized

It is intended to be imported by the main `specz.py` module.
Functions are copied verbatim from the original file to avoid behavior changes.

Public API:
    - _honor_user_homogenized_mapping
    - _homogenize
"""

# -----------------------
# Standard library
# -----------------------
import ast as _ast
import builtins
import difflib
import logging
import math

# -----------------------
# Third-party
# -----------------------
import dask
import dask.dataframe as dd
import numpy as np
import pandas as pd

# -----------------------
# Arrow-backed dtypes (pandas >= 2.x)
# (duplicated here to avoid circular imports)
# -----------------------
try:
    import pyarrow as pa  # noqa: F401
    USE_ARROW_TYPES = True
except Exception:  # pragma: no cover
    USE_ARROW_TYPES = False

if USE_ARROW_TYPES:
    if hasattr(pd, "ArrowDtype"):
        DTYPE_STR = pd.ArrowDtype(pd.ArrowDtype(pa.string()).pyarrow_dtype)
    else:
        DTYPE_STR = "string"
    DTYPE_FLOAT = pd.ArrowDtype(pa.float64())
    DTYPE_INT = pd.ArrowDtype(pa.int64())
    DTYPE_BOOL = pd.ArrowDtype(pa.bool_())
else:
    DTYPE_STR = "string"
    DTYPE_FLOAT = "Float64"
    DTYPE_INT = "Int64"
    DTYPE_BOOL = "boolean"

# -----------------------
# Constants (also used by _normalize_types in specz.py)
# -----------------------
JADES_LETTER_TO_SCORE = {"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0, "E": 0.0}
VIMOS_FLAG_TO_SCORE = {
    "LRB_X": 0.0, "MR_X": 0.0,
    "LRB_B": 1.0, "LRB_C": 1.0, "MR_C": 1.0,
    "MR_B": 3.0,
    "LRB_A": 4.0, "MR_A": 4.0,
}

_TRANSLATION_KEYS = {
    "z_flag_translation": ("float", {0.0, 1.0, 2.0, 3.0, 4.0}),
    "instrument_type_translation": ("str", {"s", "g", "p"}),
    "object_type_translation": (
        "str",
        {"star", "galactic", "qso", "agn", "galaxy"},
    ),
}
_RULE_OPTIONS = {
    "conditions", "default", "source", "optional_source",
    "allow_condition_overlap", "fast_path",
}
_TOP_LEVEL_KEYS = {
    "tiebreaking_priority", "delta_z_threshold", "crossmatch_radius_arcsec",
    "margin_threshold_arcsec", "margin_warning_fraction",
    "validate_global_graph_edges", "validate_global_tie_invariants",
    "validate_crd_id_uniqueness", "repartition_prepared_catalogs",
    "prepared_partition_size", "crossmatch_n_neighbors",
    "crossmatch_saturation_enabled", "crossmatch_saturation_warn_fraction",
    "crossmatch_saturation_fail_fraction",
    "crossmatch_geometry_diagnostics_enabled",
    "representative_radius_diagnostics_enabled", "dedup_edge_diagnostics_enabled",
    "tie_invariant_diagnostics_enabled",
    "tie_invariant_diagnostics_detailed_enabled",
    "tie_invariant_diagnostics_sample_size",
    "tie_invariant_diagnostics_max_rows",
    "label_merge_diagnostics_enabled",
    "instrument_type_priority", "save_expr_columns", "expr_column_schema",
    "runtime_schema_hints", "translation_rules",
}
_SAFE_FUNCTIONS = {"len", "int", "float", "str"}
_SAFE_NUMPY_FUNCTIONS = {"isfinite", "abs", "trunc"}
_SAFE_SERIES_METHODS = {"isin", "contains", "strip", "lower"}
_SAFE_AST_NODES = (
    _ast.Expression, _ast.BoolOp, _ast.BinOp, _ast.UnaryOp, _ast.Compare,
    _ast.Name, _ast.Load, _ast.Constant, _ast.Call, _ast.Attribute,
    _ast.Subscript, _ast.Slice, _ast.List, _ast.Tuple, _ast.keyword,
    _ast.And, _ast.Or, _ast.Not, _ast.Invert, _ast.UAdd, _ast.USub,
    _ast.Add, _ast.Sub, _ast.Mult, _ast.Div, _ast.FloorDiv, _ast.Mod,
    _ast.Pow, _ast.BitAnd, _ast.BitOr, _ast.Eq, _ast.NotEq, _ast.Lt,
    _ast.LtE, _ast.Gt, _ast.GtE, _ast.In, _ast.NotIn,
)


class _SafeExpressionValidator(_ast.NodeVisitor):
    """Reject expression syntax outside the small vectorized-rule DSL."""

    def __init__(self, path: str):
        self.path = path

    def fail(self, node: _ast.AST, detail: str) -> None:
        raise ValueError(
            f"{self.path}: forbidden expression element {type(node).__name__}: {detail}"
        )

    def generic_visit(self, node: _ast.AST) -> None:
        if not isinstance(node, _SAFE_AST_NODES):
            self.fail(node, "not allowed by the translation expression DSL")
        super().generic_visit(node)

    def visit_Name(self, node: _ast.Name) -> None:
        if node.id.startswith("_"):
            self.fail(node, f"private name {node.id!r} is not allowed")

    def visit_Attribute(self, node: _ast.Attribute) -> None:
        if node.attr.startswith("_"):
            self.fail(node, f"private attribute {node.attr!r} is not allowed")
        if isinstance(node.value, _ast.Name) and node.value.id == "np":
            if node.attr not in _SAFE_NUMPY_FUNCTIONS:
                self.fail(node, f"np.{node.attr} is not allowed")
        elif node.attr not in _SAFE_SERIES_METHODS and node.attr != "str":
            self.fail(node, f"attribute or method {node.attr!r} is not allowed")
        self.generic_visit(node)

    def visit_Call(self, node: _ast.Call) -> None:
        if isinstance(node.func, _ast.Name):
            if node.func.id not in _SAFE_FUNCTIONS:
                self.fail(node, f"function {node.func.id!r} is not allowed")
        elif isinstance(node.func, _ast.Attribute):
            self.visit_Attribute(node.func)
        else:
            self.fail(node, "only whitelisted named functions and methods are allowed")
        for arg in node.args:
            self.visit(arg)
        for keyword in node.keywords:
            if keyword.arg is None or keyword.arg.startswith("_"):
                self.fail(keyword, "keyword expansion/private keywords are not allowed")
            self.visit(keyword.value)


def _validate_expression(expr: str, path: str) -> None:
    try:
        tree = _ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"{path}: invalid expression syntax: {exc.msg}") from exc
    _SafeExpressionValidator(path).visit(tree)


def validate_translation_config(config: dict) -> None:
    """Validate user-editable translation rules before catalog processing."""
    if not isinstance(config, dict):
        raise TypeError("flags translation root must be a mapping")
    unknown_top = set(config) - _TOP_LEVEL_KEYS
    if unknown_top:
        raise ValueError(f"flags translation root: unknown option(s): {sorted(unknown_top)}")
    for key in (
        "tie_invariant_diagnostics_enabled",
        "tie_invariant_diagnostics_detailed_enabled",
        "label_merge_diagnostics_enabled",
    ):
        if key in config and not isinstance(config[key], bool):
            raise TypeError(f"{key} must be a boolean")
    for key in (
        "tie_invariant_diagnostics_sample_size",
        "tie_invariant_diagnostics_max_rows",
    ):
        if key in config and (
            not isinstance(config[key], int)
            or isinstance(config[key], bool)
            or config[key] < 1
        ):
            raise ValueError(f"{key} must be a positive integer")
    rules = config.get("translation_rules", {})
    runtime_hints = config.get("runtime_schema_hints", {})
    if not isinstance(runtime_hints, dict):
        raise TypeError("runtime_schema_hints must be a mapping")
    for column, kind in runtime_hints.items():
        if not isinstance(column, str) or not column.strip():
            raise ValueError("runtime_schema_hints column names must be non-empty strings")
        if kind not in {"str", "float", "int", "bool"}:
            raise ValueError(
                f"runtime_schema_hints.{column}={kind!r} is invalid; "
                "expected str, float, int or bool"
            )
    if not isinstance(rules, dict):
        raise TypeError("translation_rules must be a mapping")
    for survey, ruleset in rules.items():
        base = f"translation_rules.{survey}"
        if not isinstance(survey, str) or not survey.strip():
            raise ValueError("translation_rules survey names must be non-empty strings")
        if not isinstance(ruleset, dict):
            raise TypeError(f"{base} must be a mapping")
        unknown_blocks = set(ruleset) - set(_TRANSLATION_KEYS)
        if unknown_blocks:
            raise ValueError(f"{base}: unknown option(s): {sorted(unknown_blocks)}")
        for block, (kind, allowed) in _TRANSLATION_KEYS.items():
            if block not in ruleset:
                continue
            path = f"{base}.{block}"
            rule = ruleset[block]
            if not isinstance(rule, dict):
                raise TypeError(f"{path} must be a mapping")
            for option in ("optional_source", "allow_condition_overlap"):
                if option in rule and not isinstance(rule[option], bool):
                    raise TypeError(f"{path}.{option} must be a boolean")
            if "source" in rule and (
                not isinstance(rule["source"], str) or not rule["source"].strip()
            ):
                raise ValueError(f"{path}.source must be a non-empty string")
            policy = rule.get("fast_path", "auto")
            if policy not in {"auto", "disabled", "required"}:
                raise ValueError(
                    f"{path}.fast_path must be one of ['auto', 'disabled', 'required']"
                )
            conditions = rule.get("conditions", [])
            if not isinstance(conditions, list):
                raise TypeError(f"{path}.conditions must be a list")
            for index, condition in enumerate(conditions):
                cpath = f"{path}.conditions[{index}]"
                if not isinstance(condition, dict):
                    raise TypeError(f"{cpath} must be a mapping")
                unknown = set(condition) - {"expr", "value"}
                if unknown:
                    raise ValueError(f"{cpath}: unknown option(s): {sorted(unknown)}")
                if not isinstance(condition.get("expr"), str) or not condition["expr"].strip():
                    raise ValueError(f"{cpath}.expr must be a non-empty string")
                if "value" not in condition:
                    raise ValueError(f"{cpath}.value is required")
                _validate_expression(condition["expr"], f"{cpath}.expr")
            values = []
            if "default" in rule:
                values.append((f"{path}.default", rule["default"]))
            values.extend(
                (f"{path}.conditions[{i}].value", c["value"])
                for i, c in enumerate(conditions)
            )
            for key, value in rule.items():
                if key not in _RULE_OPTIONS:
                    close = difflib.get_close_matches(str(key), _RULE_OPTIONS, n=1, cutoff=0.8)
                    if close:
                        raise ValueError(
                            f"{path}.{key}: unknown option; did you mean {close[0]!r}?"
                        )
                    values.append((f"{path}.{key}", value))
                    if block == "z_flag_translation":
                        try:
                            float(key)
                        except (TypeError, ValueError) as exc:
                            raise ValueError(
                                f"{path}.{key}: z_flag direct mapping keys must be numeric"
                            ) from exc
            for value_path, value in values:
                if value is None:
                    continue
                normalized = str(value).strip().lower() if kind == "str" else value
                if kind == "float":
                    try:
                        normalized = float(value)
                    except (TypeError, ValueError) as exc:
                        raise ValueError(f"{value_path} must be numeric or null") from exc
                if normalized not in allowed:
                    raise ValueError(
                        f"{value_path}={value!r} is invalid; allowed values are "
                        f"{sorted(allowed)} or null"
                    )

# -----------------------
# Local helper (duplicated to avoid circular dep)
# -----------------------
def _normalize_string_series_to_na(s: pd.Series) -> pd.Series:
    """Normalize to string dtype and coerce placeholders to <NA>.

    Args:
        s: Input series.

    Returns:
        pd.Series: Normalized string series.
    """
    s = s.astype(DTYPE_STR).str.strip()
    low = s.fillna("").str.lower()
    mask_empty = (low == "") | low.isin(["none", "null", "nan"])
    return s.mask(mask_empty, pd.NA).astype(DTYPE_STR)


# -----------------------
# Public API
# -----------------------
def _honor_user_homogenized_mapping(
    df: dd.DataFrame,
    entry: dict,
    product_name: str,
    logger: logging.Logger,
) -> dd.DataFrame:
    """Mirror user-provided homogenized columns declared in YAML.

    Args:
        df: Frame after YAML rename.
        entry: YAML node.
        product_name: Catalog identifier.
        logger: Logger.

    Returns:
        dd.DataFrame: Frame with canonical homogenized columns mirrored.
    """
    def _ensure_single_series(df: dd.DataFrame, name: str):
        if list(df.columns).count(name) > 1:
            raise ValueError(
                f"[{product_name}] Duplicate '{name}' after YAML rename; "
                "ensure unique column names."
            )
        return df[name]

    cols_cfg = (entry.get("columns") or {})
    norm = lambda x: (str(x).strip().lower() if x is not None else None)

    if norm(cols_cfg.get("z_flag")) == "z_flag_homogenized":
        if "z_flag" in df.columns:
            df["z_flag_homogenized"] = _ensure_single_series(df, "z_flag")
            logger.info(f"{product_name} Using user-provided homogenized z_flag via YAML mapping.")
        else:
            logger.warning(f"{product_name} YAML says z_flag<-z_flag_homogenized, but 'z_flag' missing.")

    if norm(cols_cfg.get("instrument_type")) == "instrument_type_homogenized":
        if "instrument_type" in df.columns:
            df["instrument_type_homogenized"] = _ensure_single_series(df, "instrument_type")
            logger.info(f"{product_name} Using user-provided homogenized instrument_type via YAML mapping.")
        else:
            logger.warning(f"{product_name} YAML says instrument_type<-instrument_type_homogenized, but missing.")

    if norm(cols_cfg.get("object_type")) == "object_type_homogenized":
        if "object_type" in df.columns:
            df["object_type_homogenized"] = _ensure_single_series(df, "object_type")
            logger.info(
                f"{product_name} Using user-provided homogenized object_type via YAML mapping."
            )
        else:
            logger.warning(
                f"{product_name} YAML says object_type<-object_type_homogenized, but missing."
            )

    return df


def _assert_yaml_coverage_for_surveys(
    df: dd.DataFrame,
    key: str,
    translation_rules_uc: dict,
    product_name: str,
    logger: logging.Logger,
) -> None:
    """Ensure that every survey present in df has a YAML translation block for `key`.

    Args:
        df: Input frame (must contain 'survey').
        key: Either 'z_flag' or 'instrument_type' - used to look up '<key>_translation'.
        translation_rules_uc: Upper-cased translation rules from YAML (survey -> ruleset).
        product_name: Catalog identifier (for error context).
        logger: Logger.

    Raises:
        ValueError: If any present survey is missing in YAML for the given key.
    """
    if "survey" not in df.columns:
        # Nothing to check; downstream logic relies on 'survey', but we won't raise here.
        logger.warning(f"{product_name} YAML translation requested but 'survey' column is missing.")
        return

    # Collect upper-cased surveys present in the data
    try:
        surveys_uc = (
            df["survey"]
            .dropna()
            .map_partitions(lambda s: s.astype(str).str.upper(),
                            meta=pd.Series(pd.array([], dtype=DTYPE_STR)))
            .unique()
            .compute()
        )
        surveys_uc = [s for s in surveys_uc if isinstance(s, (str, np.str_)) and s.strip() != ""]
    except Exception as e:
        logger.warning(f"{product_name} Could not extract unique surveys for YAML coverage check: {e}")
        return

    # Surveys that have a translation block for this key in YAML
    key_block = f"{key}_translation"
    yaml_surveys_with_key = {
        sname for sname, ruleset in translation_rules_uc.items()
        if isinstance(ruleset.get(key_block), dict)  # a dict (possibly with default/conditions/direct maps)
    }

    missing = sorted(set(surveys_uc) - yaml_surveys_with_key)
    if missing:
        available = sorted(yaml_surveys_with_key)
        raise ValueError(
            f"[{product_name}] Missing YAML translation for '{key}' in surveys: {missing}. "
            f"Add blocks under translation_rules.<SURVEY>.{key_block}. "
            f"Currently available (for '{key}'): {available}"
        )


def _homogenize(
    df: dd.DataFrame,
    translation_config: dict,
    product_name: str,
    logger: logging.Logger,
    type_cast_ok: bool,
    *,
    require_z_flag_homogenized: bool = False,
) -> tuple[dd.DataFrame, bool, list, dict, dict]:
    """Compute homogenized columns for tie-breaking.

    Args:
        df: Frame after type normalization.
        translation_config: Config with priorities and rules.
        product_name: Catalog identifier.
        logger: Logger.
        type_cast_ok: Whether `type` was normalized.
        require_z_flag_homogenized: Create and validate the quality flag even
            when it is not a ranking priority.

    Returns:
        Tuple: (df, used_type_fastpath, tiebreaking_priority, instrument_type_priority, translation_rules_uc)
    """
    validate_translation_config(translation_config)
    tiebreaking_priority = translation_config.get("tiebreaking_priority", [])
    instrument_type_priority = translation_config.get("instrument_type_priority", {})
    translation_rules_uc = {k.upper(): v for k, v in translation_config.get("translation_rules", {}).items()}
    validated_non_null_counts: dict[str, int] = {}
    z_flag_is_priority = "z_flag_homogenized" in tiebreaking_priority
    needs_z_flag = z_flag_is_priority or require_z_flag_homogenized

    def _fast_path_policy(key: str) -> str:
        if "survey" not in df.columns:
            return "auto"
        surveys = (
            df["survey"].dropna().astype(str).str.upper().unique().compute().tolist()
        )
        policies = {
            str(
                translation_rules_uc.get(survey, {})
                .get(f"{key}_translation", {})
                .get("fast_path", "auto")
            )
            for survey in surveys
        }
        if "disabled" in policies and "required" in policies:
            raise ValueError(
                f"[{product_name}] Conflicting {key} fast_path policies for surveys "
                f"{sorted(surveys)}: cannot mix 'disabled' and 'required' in one catalog"
            )
        if "disabled" in policies:
            return "disabled"
        if "required" in policies:
            return "required"
        return "auto"

    def _validate_result_domain(
        column: str, allowed: set, *, normalize_string: bool = False
    ) -> None:
        values = df[column]
        if normalize_string:
            values = values.map_partitions(
                _normalize_string_series_to_na,
                meta=pd.Series(pd.array([], dtype=DTYPE_STR)),
            ).str.lower()
        else:
            values = dd.to_numeric(values, errors="coerce")
        invalid = (~dd.isna(values)) & ~values.isin(list(allowed))
        invalid_count, non_null_count = dask.compute(invalid.sum(), values.count())
        validated_non_null_counts[column] = int(non_null_count)
        if int(invalid_count):
            examples = df[column].loc[invalid].head(5, compute=True).tolist()
            raise ValueError(
                f"[{product_name}] Translation produced invalid values in '{column}'. "
                f"Allowed set is {sorted(allowed)} (NaN allowed). Examples: {examples}"
            )

    # -----------------------
    # Vectorized translator
    # -----------------------
    def _translate_column_vectorized(df: dd.DataFrame, key: str, out_col: str, out_kind: str) -> dd.DataFrame:
        """Apply YAML translation rules per partition."""
        assert key in {"z_flag", "instrument_type", "object_type"}
        assert out_col in {
            "z_flag_homogenized",
            "instrument_type_homogenized",
            "object_type_homogenized",
        }
        assert out_kind in {"float", "str"}

        def _partition(p: pd.DataFrame) -> pd.DataFrame:
            if p.empty or ("survey" not in p.columns) or (key not in p.columns):
                q = p.copy()
                if out_kind == "float":
                    q[out_col] = pd.Series(pd.array([], dtype=DTYPE_FLOAT)).reindex(q.index)
                else:
                    q[out_col] = pd.Series(pd.array([], dtype=DTYPE_STR)).reindex(q.index)
                return q

            s = p.copy()
            survey_uc = s["survey"].astype(str).str.upper()

            if out_kind == "float":
                out = pd.Series(pd.array([pd.NA] * len(s), dtype=DTYPE_FLOAT), index=s.index)
            else:
                out = pd.Series(pd.array([pd.NA] * len(s), dtype=DTYPE_STR), index=s.index)

            class StrSeriesProxy:
                def __init__(self, ser: pd.Series):
                    self._s = ser.astype(DTYPE_STR)
                def __getitem__(self, key):
                    if isinstance(key, slice):
                        return self._s.str.slice(key.start, key.stop, key.step)
                    return self._s.str.get(key)

            def v_len(x):
                if isinstance(x, StrSeriesProxy):
                    return x._s.str.len()
                if isinstance(x, pd.Series):
                    return x.astype(DTYPE_STR).str.len()
                return builtins.len(x)

            def v_str(x):
                if isinstance(x, StrSeriesProxy):
                    return x
                if isinstance(x, pd.Series):
                    return StrSeriesProxy(x)
                return builtins.str(x)

            def v_int(x):
                if isinstance(x, StrSeriesProxy):
                    x = x._s
                if isinstance(x, pd.Series):
                    return pd.to_numeric(x, errors="coerce").astype(DTYPE_INT)
                return builtins.int(x)

            def v_float(x):
                if isinstance(x, StrSeriesProxy):
                    x = x._s
                if isinstance(x, pd.Series):
                    return pd.to_numeric(x, errors="coerce").astype(DTYPE_FLOAT)
                return builtins.float(x)

            class _BoolToBitwise(_ast.NodeTransformer):
                def visit_BoolOp(self, node):
                    self.generic_visit(node)
                    op = _ast.BitAnd() if isinstance(node.op, _ast.And) else _ast.BitOr()
                    expr = node.values[0]
                    for v in node.values[1:]:
                        expr = _ast.BinOp(left=expr, op=op, right=v)
                    return expr
                def visit_UnaryOp(self, node):
                    self.generic_visit(node)
                    if isinstance(node.op, _ast.Not):
                        return _ast.UnaryOp(op=_ast.Invert(), operand=node.operand)
                    return node
                def visit_Compare(self, node):
                    self.generic_visit(node)
                    if len(node.ops) <= 1:
                        return node
                    expr = _ast.Compare(left=node.left, ops=[node.ops[0]], comparators=[node.comparators[0]])
                    for i in range(1, len(node.ops)):
                        cmp_i = _ast.Compare(left=node.comparators[i-1], ops=[node.ops[i]], comparators=[node.comparators[i]])
                        expr = _ast.BinOp(left=expr, op=_ast.BitAnd(), right=cmp_i)
                    return expr

            for sname, ruleset in translation_rules_uc.items():
                rule = (ruleset.get(f"{key}_translation") or {})
                if not isinstance(rule, dict) or not len(rule):
                    continue

                mask_s = (survey_uc == sname)
                if not mask_s.any():
                    continue

                default_val = rule.get("default", (np.nan if out_kind == "float" else ""))

                if out_kind == "float":
                    out.loc[mask_s] = out.loc[mask_s].fillna(default_val)
                else:
                    fill_vals = pd.Series([default_val] * int(mask_s.sum()), index=out.index[mask_s], dtype=DTYPE_STR)
                    out.loc[mask_s] = out.loc[mask_s].fillna(fill_vals)

                source_col = str(rule.get("source", key))
                optional_source = bool(rule.get("optional_source", False))
                allow_condition_overlap = bool(
                    rule.get("allow_condition_overlap", False)
                )
                direct = {
                    k: v
                    for k, v in rule.items()
                    if k not in {
                        "conditions",
                        "default",
                        "source",
                        "optional_source",
                        "allow_condition_overlap",
                        "fast_path",
                    }
                }
                if direct:
                    if source_col not in s.columns:
                        if not optional_source:
                            raise ValueError(
                                f"Missing source column '{source_col}' for survey "
                                f"'{sname}' and translation '{out_col}'."
                            )
                    else:
                        col = s.loc[mask_s, source_col]
                        is_num = pd.api.types.is_numeric_dtype(col)
                        if key == "z_flag":
                            is_num = True
                        if is_num:
                            col_num = pd.to_numeric(col, errors="coerce")
                            num_map = {float(rk): rv for rk, rv in direct.items()}
                            mapped = col_num.map(num_map)
                        else:
                            col_str = col.astype(str).str.strip().str.lower()
                            str_map = {str(k).strip().lower(): v for k, v in direct.items()}
                            mapped = col_str.map(str_map)

                        if out_kind == "float":
                            out.loc[mask_s] = mapped.fillna(out.loc[mask_s]).astype(DTYPE_FLOAT)
                        else:
                            mapped = mapped.astype("object")
                            mapped_str = pd.Series(pd.array(mapped.where(mapped.notna(), None), dtype=DTYPE_STR), index=mapped.index)
                            take = mapped_str.notna()
                            out.loc[take.index] = out.loc[take.index].where(~take, mapped_str)

                condition_values = pd.Series(
                    pd.array([pd.NA] * len(s), dtype="object"), index=s.index
                )
                for condition_index, cond in enumerate(
                    rule.get("conditions") or []
                ):
                    expr = cond.get("expr")
                    if not expr:
                        continue

                    ctx = {c: s[c] for c in s.columns if c in s.columns}
                    safe_globals = {
                        "__builtins__": {},
                        "len": v_len,
                        "int": v_int,
                        "float": v_float,
                        "str": v_str,
                        "math": math,
                        "np": np,
                        "pd": pd,
                    }

                    try:
                        tree = _ast.parse(expr, mode="eval")
                        tree2 = _BoolToBitwise().visit(tree)
                        _ast.fix_missing_locations(tree2)
                        expr_vec = _ast.unparse(tree2)
                        mlocal = eval(expr_vec, safe_globals, ctx)
                    except Exception as e:
                        raise ValueError(
                            f"Error evaluating condition '{expr}' for survey "
                            f"'{sname}': {e}"
                        ) from e

                    if isinstance(mlocal, pd.Series):
                        mlocal = mlocal.reindex(s.index)
                    elif hasattr(mlocal, "__len__") and not np.isscalar(mlocal) and len(mlocal) == len(s):
                        mlocal = pd.Series(mlocal, index=s.index)
                    else:
                        mlocal = pd.Series(bool(mlocal), index=s.index)

                    mlocal = mlocal.astype("boolean").fillna(False)
                    mask_s_aligned = mask_s.reindex(s.index).fillna(False).astype(bool)
                    mlocal = mlocal & mask_s_aligned

                    val = cond.get("value", default_val)
                    normalized_val = pd.NA if pd.isna(val) else val
                    conflicting = (
                        mlocal
                        & condition_values.notna()
                        & condition_values.ne(normalized_val).fillna(False)
                    )
                    conflict_count = int(conflicting.sum())
                    if conflict_count and not allow_condition_overlap:
                        logger.warning(
                            "[%s] %s.%s condition %d overwrites a different "
                            "condition value for %d row(s); later conditions "
                            "take precedence. Set allow_condition_overlap: true "
                            "when this is intentional.",
                            product_name,
                            sname,
                            f"{key}_translation",
                            condition_index + 1,
                            conflict_count,
                        )
                    condition_values.loc[mlocal] = normalized_val
                    if out_kind == "float":
                        out.loc[mlocal] = pd.to_numeric(val, errors="coerce")
                    else:
                        out.loc[mlocal] = (pd.NA if pd.isna(val) else str(val))

            if out_col == "z_flag_homogenized":
                s[out_col] = pd.to_numeric(out, errors="coerce").astype(DTYPE_FLOAT)
            else:
                s[out_col] = pd.Series(out, dtype=DTYPE_STR).str.lower()
            return s

        meta = df._meta.copy()
        if out_kind == "float":
            meta[out_col] = pd.Series(pd.array([], dtype=DTYPE_FLOAT))
        else:
            meta[out_col] = pd.Series(pd.array([], dtype=DTYPE_STR))
        return df.map_partitions(_partition, meta=meta)

    # z_flag_homogenized
    def can_use_zflag_as_quality() -> bool:
        if "z_flag" not in df.columns:
            return False
        try:
            cnt = df["z_flag"].count()
            minv = df["z_flag"].min()
            maxv = df["z_flag"].max()
            frac_cnt = df[(df["z_flag"] > 0.0) & (df["z_flag"] < 1.0)]["z_flag"].count()
            cnt, minv, maxv, frac_cnt = dask.compute(cnt, minv, maxv, frac_cnt)
            if cnt == 0 or not np.isfinite(minv) or not np.isfinite(maxv):
                return False
            if not (minv >= 0.0 and maxv <= 1.0):
                return False
            return frac_cnt > 0
        except Exception as e:
            logger.warning(f"{product_name} Could not validate 'z_flag' as quality-like: {e}")
            return False

    def quality_like_to_flag(x):
        if pd.isna(x):
            return np.nan
        x = float(x)
        if x == 0.0:
            return 0.0
        if 0.0 < x < 0.7:
            return 1.0
        if 0.7 <= x < 0.9:
            return 2.0
        if 0.9 <= x < 0.99:
            return 3.0
        if 0.99 <= x <= 1.0:
            return 4.0
        return np.nan

    if needs_z_flag:
        if "z_flag_homogenized" not in df.columns:
            z_fast_path = _fast_path_policy("z_flag")
            z_fast_path_compatible = can_use_zflag_as_quality()
            if z_fast_path == "required" and not z_fast_path_compatible:
                raise ValueError(
                    f"[{product_name}] z_flag fast_path is 'required', but z_flag "
                    "is not a non-empty probability-like column in [0, 1]"
                )
            if z_fast_path != "disabled" and z_fast_path_compatible:
                logger.info(
                    "%s Using 'z_flag' fast path for z_flag_homogenized "
                    "(policy=%s); YAML z_flag rules are bypassed.",
                    product_name,
                    z_fast_path,
                )
                df["z_flag_homogenized"] = df["z_flag"].map_partitions(
                    lambda s: s.apply(quality_like_to_flag).astype(DTYPE_FLOAT),
                    meta=pd.Series(pd.array([], dtype=DTYPE_FLOAT)),
                )
            else:
                logger.info(f"{product_name} Using YAML translation for z_flag_homogenized.")
                # NEW: assert that all surveys present have YAML coverage for z_flag
                _assert_yaml_coverage_for_surveys(
                    df=df,
                    key="z_flag",
                    translation_rules_uc=translation_rules_uc,
                    product_name=product_name,
                    logger=logger,
                )
                df = _translate_column_vectorized(df, key="z_flag",
                                                  out_col="z_flag_homogenized",
                                                  out_kind="float")
                _validate_result_domain(
                    "z_flag_homogenized", {0.0, 1.0, 2.0, 3.0, 4.0}
                )
        else:
            # User-provided 'z_flag_homogenized' is present. Validate allowed domain {0,1,2,3,4} (NaN allowed).
            logger.info(f"{product_name} 'z_flag_homogenized' already exists; validating user-provided values.")
            allowed = {0.0, 1.0, 2.0, 3.0, 4.0}

            vals = dd.to_numeric(df["z_flag_homogenized"], errors="coerce")
            # NaN is allowed; only non-NaN values outside the allowed set are invalid
            invalid_mask = (
                ((~dd.isna(df["z_flag_homogenized"])) & dd.isna(vals))
                | ((~dd.isna(vals)) & ~vals.isin(list(allowed)))
            )
            invalid_count, non_null_count = dask.compute(
                invalid_mask.sum(), vals.count()
            )
            validated_non_null_counts["z_flag_homogenized"] = int(non_null_count)

            if invalid_count > 0:
                examples = df["z_flag_homogenized"].loc[invalid_mask].head(5, compute=True).tolist()
                raise ValueError(
                    f"[{product_name}] Invalid values in user-provided 'z_flag_homogenized'. "
                    f"Allowed set is {sorted(allowed)} (NaN allowed). Examples of invalid values: {examples}"
                )

            # Cast to Arrow-backed float dtype for consistency
            df["z_flag_homogenized"] = vals.map_partitions(
                lambda s: s.astype(DTYPE_FLOAT),
                meta=pd.Series(pd.array([], dtype=DTYPE_FLOAT)),
            )


    # instrument_type_homogenized
    used_type_fastpath = False

    def can_use_type_for_instrument() -> bool:
        if not type_cast_ok or "type" not in df.columns:
            return False
        try:
            allowed = {"s", "g", "p"}
            uniques = df["type"].dropna().unique().compute().tolist()
            uniques = [u for u in uniques if isinstance(u, (str, np.str_)) and u != ""]
            if len(uniques) == 0:
                return False
            return all(u in allowed for u in uniques)
        except Exception as e:
            logger.warning(f"{product_name} Could not validate 'type' values: {e}")
            return False

    if "instrument_type_homogenized" in tiebreaking_priority:
        if "instrument_type_homogenized" not in df.columns:
            instrument_fast_path = _fast_path_policy("instrument_type")
            instrument_fast_path_compatible = can_use_type_for_instrument()
            if instrument_fast_path == "required" and not instrument_fast_path_compatible:
                raise ValueError(
                    f"[{product_name}] instrument_type fast_path is 'required', "
                    "but type is not a non-empty column containing only s/g/p"
                )
            if instrument_fast_path != "disabled" and instrument_fast_path_compatible:
                logger.info(
                    "%s Using 'type' fast path for instrument_type_homogenized "
                    "(policy=%s); YAML instrument rules are bypassed.",
                    product_name,
                    instrument_fast_path,
                )
                df["instrument_type_homogenized"] = df["type"].map_partitions(
                    _normalize_string_series_to_na,
                    meta=pd.Series(pd.array([], dtype=DTYPE_STR)),
                ).str.lower()
                used_type_fastpath = True
            else:
                logger.info(f"{product_name} Using YAML translation for instrument_type_homogenized.")
                # NEW: assert that all surveys present have YAML coverage for instrument_type
                _assert_yaml_coverage_for_surveys(
                    df=df,
                    key="instrument_type",
                    translation_rules_uc=translation_rules_uc,
                    product_name=product_name,
                    logger=logger,
                )
                df = _translate_column_vectorized(df, key="instrument_type",
                                                  out_col="instrument_type_homogenized",
                                                  out_kind="str")
                df["instrument_type_homogenized"] = df["instrument_type_homogenized"].map_partitions(
                    _normalize_string_series_to_na,
                    meta=pd.Series(pd.array([], dtype=DTYPE_STR)),
                ).str.lower()
                _validate_result_domain(
                    "instrument_type_homogenized",
                    {"s", "g", "p"},
                    normalize_string=True,
                )
        else:
            # User-provided 'instrument_type_homogenized' is present. Validate allowed domain {"s","p","g"}.
            logger.info(f"{product_name} 'instrument_type_homogenized' already exists; validating user-provided values.")
            allowed = {"s", "p", "g"}

            normed = df["instrument_type_homogenized"].map_partitions(
                _normalize_string_series_to_na,
                meta=pd.Series(pd.array([], dtype=DTYPE_STR)),
            ).str.lower()

            invalid_mask = (~dd.isna(normed)) & ~normed.isin(list(allowed))
            invalid_count, non_null_count = dask.compute(
                invalid_mask.sum(), normed.count()
            )
            validated_non_null_counts["instrument_type_homogenized"] = int(
                non_null_count
            )

            if invalid_count > 0:
                examples = df["instrument_type_homogenized"].loc[invalid_mask].head(5, compute=True).tolist()
                raise ValueError(
                    f"[{product_name}] Invalid values in user-provided 'instrument_type_homogenized'. "
                    f"Allowed set is {sorted(allowed)}. Examples of invalid values: {examples}"
                )

            # Keep normalized lower-case values for consistency
            df["instrument_type_homogenized"] = normed

    # object_type_homogenized is an output-schema field, not a ranking field.
    # An entirely-null result is valid for catalogs without classification data.
    if "object_type_homogenized" not in df.columns:
        df = _translate_column_vectorized(
            df,
            key="object_type",
            out_col="object_type_homogenized",
            out_kind="str",
        )

    object_types = df["object_type_homogenized"].map_partitions(
        _normalize_string_series_to_na,
        meta=pd.Series(pd.array([], dtype=DTYPE_STR)),
    ).str.lower()
    allowed_object_types = {"star", "galactic", "qso", "agn", "galaxy"}
    invalid_object_mask = (~dd.isna(object_types)) & ~object_types.isin(
        list(allowed_object_types)
    )
    invalid_object_count = int(invalid_object_mask.sum().compute())
    if invalid_object_count:
        examples = (
            df["object_type_homogenized"]
            .loc[invalid_object_mask]
            .head(5, compute=True)
            .tolist()
        )
        raise ValueError(
            f"[{product_name}] Invalid values in 'object_type_homogenized'. "
            f"Allowed set is {sorted(allowed_object_types)} (NaN allowed). "
            f"Examples of invalid values: {examples}"
        )
    df["object_type_homogenized"] = object_types

    # --- post-homogenization sanity checks (required columns must not be all-NaN) ---
    if needs_z_flag:
        if "z_flag_homogenized" not in df.columns:
            raise ValueError(
                f"[{product_name}] 'z_flag_homogenized' is required for "
                "quality ranking or filtering but is missing after homogenization."
            )
        # An all-null quality column is valid: this priority simply cannot
        # distinguish candidates. Later priorities or hard-tie handling apply.

    if "instrument_type_homogenized" in tiebreaking_priority:
        if "instrument_type_homogenized" not in df.columns:
            raise ValueError(
                f"[{product_name}] 'instrument_type_homogenized' is required by tiebreaking_priority but is missing after homogenization."
            )
        non_null = validated_non_null_counts.get("instrument_type_homogenized")
        if non_null is None:
            non_null = dask.compute(df["instrument_type_homogenized"].count())[0]
        if int(non_null) == 0:
            raise ValueError(
                f"[{product_name}] All values in 'instrument_type_homogenized' are NaN. "
                "This column is required (in tiebreaking_priority) and must contain at least one non-NaN value. "
                "Verify YAML translations / fast-path logic and input columns."
            )

    return df, used_type_fastpath, tiebreaking_priority, instrument_type_priority, translation_rules_uc
