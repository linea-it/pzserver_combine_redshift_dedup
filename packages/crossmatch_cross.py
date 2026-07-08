from __future__ import annotations

"""Crossmatch and `compared_to` updater for CRC.

This module supports two export backends, controlled by the hardcoded flag
`USE_LSDB_CONCAT`.

Backends:
    - If USE_LSDB_CONCAT = True (default): run the crossmatch, update
      `compared_to`, normalize dtypes per catalog, concatenate at the LSDB
      catalog level, and persist a HATS collection via `write_catalog`.
      Returns the collection path.
    - If USE_LSDB_CONCAT = False: run the crossmatch, update `compared_to`,
      concatenate the underlying Dask DataFrames, normalize dtypes on the
      merged Dask DataFrame (expected and expr), write a Parquet dataset,
      and import it as an LSDB collection (margin-first retry).
      Returns the collection path.

Public API:
    - crossmatch_tiebreak
    - crossmatch_tiebreak_safe
"""

# -----------------------
# Backend switch
# -----------------------
USE_LSDB_CONCAT: bool = True  # Toggle behavior as described above.
BACKEND_LSDB_LABEL = "LSDB+write_catalog"
BACKEND_LEGACY_LABEL = "Dask+Parquet+import"
BACKEND_DISTRIBUTED_PAIRS_LABEL = "distributed-pairs+Parquet+import"

# -----------------------
# Standard library
# -----------------------
import logging
import os
import shutil
import time
from typing import Dict, Iterable, List, Set

import dask
import dask.dataframe as dd

# -----------------------
# Third-party
# -----------------------
import numpy as np
import pandas as pd
from crossmatch_diagnostics import (
    log_component_size_diagnostics,
    log_neighbor_count_diagnostics,
    log_pair_separation_diagnostics,
    project_catalog_for_pair_crossmatch,
    stage_projected_pairs,
)
from specz import (
    DTYPE_BOOL,
    DTYPE_FLOAT,
    DTYPE_INT,
    DTYPE_INT8,
    DTYPE_STR,
    _add_missing_with_dtype,
    _build_collection_with_retry,
    _normalize_string_series_to_na,
)

# -----------------------
# Project
# -----------------------
from utils import get_phase_logger

__all__ = ["crossmatch_tiebreak", "crossmatch_tiebreak_safe"]

LOGGER_NAME = "crc.crossmatch"  # child of the pipeline root logger ("crc")


# -----------------------
# Centralized logging
# -----------------------
def _get_logger() -> logging.LoggerAdapter:
    """Return a phase-aware logger ('crc.crossmatch' with phase='crossmatch').

    Returns:
        logging.LoggerAdapter: Logger with phase context.
    """
    base = logging.getLogger(LOGGER_NAME)
    base.setLevel(logging.NOTSET)
    base.propagate = True
    return get_phase_logger("crossmatch", base)


def _get_backend_label() -> str:
    """Return the active backend label for logging.

    Returns:
        str: Active backend label.
    """
    return BACKEND_DISTRIBUTED_PAIRS_LABEL


# -----------------------
# Utilities
# -----------------------
def _adjacency_from_pairs(
    left_ids: pd.Series, right_ids: pd.Series
) -> Dict[str, Set[str]]:
    """Build an undirected adjacency from left-right crossmatch pairs.

    Args:
        left_ids: Left-side CRD_ID values.
        right_ids: Right-side CRD_ID values.

    Returns:
        Dict[str, Set[str]]: Mapping of node id to neighbor ids.
    """
    adj: Dict[str, Set[str]] = {}
    L = left_ids.astype(str).to_numpy(dtype=object, copy=False)
    R = right_ids.astype(str).to_numpy(dtype=object, copy=False)
    get = adj.get
    for a, b in zip(L, R):
        if a == b:
            continue
        s = get(a)
        if s is None:
            adj[a] = {b}
        else:
            s.add(b)
        s = get(b)
        if s is None:
            adj[b] = {a}
        else:
            s.add(a)
    return adj


def _log_neighbor_saturation(
    pairs: pd.DataFrame,
    *,
    id_col: str,
    source_col: str | None,
    limit: int,
    logger: logging.LoggerAdapter,
    context: str,
    total_by_source: dict,
    warn_fraction: float,
    fail_fraction: float | None,
) -> None:
    """Apply logging/failure policy to objects reaching the neighbor cap."""
    if pairs.empty or limit <= 0:
        return
    columns = [id_col, "CRD_IDright"] + (
        [source_col] if source_col and source_col in pairs else []
    )
    unique_pairs = pairs[columns].drop_duplicates()
    if source_col and source_col in unique_pairs:
        counts = unique_pairs.groupby([source_col, id_col], dropna=False).size()
        for source, source_counts in counts.groupby(level=0, dropna=False):
            log_neighbor_count_diagnostics(
                source_counts,
                limit=limit,
                logger=logger,
                context=f"{context} source={source}",
            )
            total = int(total_by_source.get(str(source), source_counts.size))
            saturated = int(source_counts.ge(limit).sum())
            fraction = saturated / total if total else 0.0
            log_method = logger.warning if fraction >= warn_fraction else logger.info
            log_method(
                "%s neighbor saturation: source=%s at_limit=%d total_objects=%d fraction=%.6f limit=%d",
                context,
                source,
                saturated,
                total,
                fraction,
                limit,
            )
            if fail_fraction is not None and fraction >= fail_fraction:
                raise RuntimeError(
                    f"{context}: neighbor saturation for source={source} is "
                    f"{fraction:.6f}, reaching failure threshold {fail_fraction:.6f}; "
                    "increase crossmatch_n_neighbors"
                )
    else:
        counts = unique_pairs.groupby(id_col).size()
        log_neighbor_count_diagnostics(
            counts,
            limit=limit,
            logger=logger,
            context=context,
        )
        total = int(total_by_source.get("<all>", counts.size))
        saturated = int(counts.ge(limit).sum())
        fraction = saturated / total if total else 0.0
        log_method = logger.warning if fraction >= warn_fraction else logger.info
        log_method(
            "%s neighbor saturation: at_limit=%d total_objects=%d fraction=%.6f limit=%d",
            context,
            saturated,
            total,
            fraction,
            limit,
        )
        if fail_fraction is not None and fraction >= fail_fraction:
            raise RuntimeError(
                f"{context}: neighbor saturation is {fraction:.6f}, reaching "
                f"failure threshold {fail_fraction:.6f}; increase crossmatch_n_neighbors"
            )


def _catalog_source_totals(catalog) -> dict:
    """Count all catalog objects by source for saturation denominators."""
    if "source" in catalog._ddf.columns:
        counts = catalog._ddf.groupby("source").size().compute()
        return {str(source): int(count) for source, count in counts.items()}
    return {"<all>": int(catalog._ddf.map_partitions(len).sum().compute())}


def _log_neighbor_saturation_distributed(
    pairs,
    *,
    id_col: str,
    source_col: str | None,
    limit: int,
    logger: logging.LoggerAdapter,
    context: str,
    total_by_source: dict,
    warn_fraction: float,
    fail_fraction: float | None,
) -> None:
    """Log saturation summaries without collecting pair rows on the driver."""
    group_cols = [id_col]
    sources: list[object] = [None]
    if source_col and source_col in pairs.columns:
        group_cols.insert(0, source_col)
        sources = pairs[source_col].dropna().drop_duplicates().compute().tolist()

    counts = pairs.groupby(group_cols).size().rename("count").to_frame().reset_index()
    for source in sources:
        source_counts = counts["count"]
        label = context
        total_key = "<all>"
        if source is not None and source_col is not None:
            source_counts = counts.loc[counts[source_col] == source, "count"]
            label = f"{context} source={source}"
            total_key = str(source)

        count, maximum, p50, p90, p99, ge2, ge5, saturated = dask.compute(
            source_counts.count(),
            source_counts.max(),
            source_counts.quantile(0.50),
            source_counts.quantile(0.90),
            source_counts.quantile(0.99),
            source_counts.ge(2).sum(),
            source_counts.ge(5).sum(),
            source_counts.ge(limit).sum(),
        )
        count = int(count)
        saturated = int(saturated)
        total = int(total_by_source.get(total_key, count))
        fraction = saturated / total if total else 0.0
        logger.info(
            "%s returned-match diagnostics: objects_with_matches=%d "
            "p50=%.1f p90=%.1f p99=%.1f max=%d fraction_ge_2=%.6f "
            "fraction_ge_5=%.6f fraction_at_limit=%.6f limit=%d",
            label,
            count,
            float(p50) if count else 0.0,
            float(p90) if count else 0.0,
            float(p99) if count else 0.0,
            int(maximum) if count else 0,
            int(ge2) / count if count else 0.0,
            int(ge5) / count if count else 0.0,
            fraction,
            limit,
        )
        log_method = logger.warning if fraction >= warn_fraction else logger.info
        log_method(
            "%s neighbor saturation: at_limit=%d total_objects=%d "
            "fraction=%.6f limit=%d",
            label,
            saturated,
            total,
            fraction,
            limit,
        )
        if fail_fraction is not None and fraction >= fail_fraction:
            raise RuntimeError(
                f"{label}: neighbor saturation is {fraction:.6f}, reaching "
                f"failure threshold {fail_fraction:.6f}; increase "
                "crossmatch_n_neighbors"
            )


def _merge_compared_to_partition(
    part: pd.DataFrame,
    pairs_adj: Dict[str, Iterable[str]],
) -> pd.DataFrame:
    """Update `compared_to` on a pandas partition.

    Args:
        part: Partition to update.
        pairs_adj: Mapping of CRD_ID to neighbor ids from crossmatch pairs.

    Returns:
        pd.DataFrame: Updated partition with merged `compared_to`.
    """
    p = part.copy()

    if "compared_to" not in p.columns:
        p["compared_to"] = pd.Series(
            pd.array([pd.NA] * len(p), dtype=DTYPE_STR), index=p.index
        )

    crd_list: List[str] = p["CRD_ID"].astype(str).tolist()

    def _norm_token(x) -> str | None:
        if pd.isna(x):
            return None
        if isinstance(x, (bool, np.bool_)):
            return None
        s = str(x).strip()
        if not s or s == "<NA>":
            return None
        return s

    def _to_str_set(it: Iterable) -> Set[str]:
        out: Set[str] = set()
        if it is None:
            return out
        for x in it:
            s = _norm_token(x)
            if s is not None:
                out.add(s)
        return out

    def _parse_existing(val) -> Set[str]:
        if isinstance(val, str):
            return _to_str_set(t.strip() for t in val.split(","))
        if isinstance(val, (list, set, tuple)):
            return _to_str_set(val)
        if pd.isna(val):
            return set()
        return _to_str_set([val])

    # Build NEW neighbor sets, parse OLD cells, then union per row.
    new_sets: List[Set[str]] = [_to_str_set(pairs_adj.get(k, ())) for k in crd_list]
    old_sets: List[Set[str]] = [_parse_existing(v) for v in p["compared_to"].tolist()]

    merged_vals: List[object] = []
    for k, old_set, new_set in zip(crd_list, old_sets, new_sets):
        nxt = set().union(old_set, new_set)
        nxt.discard(k)
        merged_vals.append(", ".join(sorted(nxt)) if nxt else pd.NA)

    p["compared_to"] = pd.Series(pd.array(merged_vals, dtype=DTYPE_STR), index=p.index)
    return p


def _merge_compared_to_column_partition(
    part: pd.DataFrame,
    new_column: str = "_new_compared_to",
) -> pd.DataFrame:
    """Union an aggregated neighbor column into ``compared_to``."""
    p = part.copy()
    if "compared_to" not in p.columns:
        p["compared_to"] = pd.Series(
            pd.array([pd.NA] * len(p), dtype=DTYPE_STR), index=p.index
        )

    def _tokens(value) -> set[str]:
        if isinstance(value, (list, set, tuple)):
            values = value
        else:
            if pd.isna(value):
                return set()
            values = str(value).split(",")
        return {
            token
            for raw in values
            if (token := str(raw).strip()) and token != "<NA>"
        }

    merged_values: list[object] = []
    for crd_id, old_value, new_value in zip(
        p["CRD_ID"].astype(str), p["compared_to"], p[new_column]
    ):
        neighbors = _tokens(old_value) | _tokens(new_value)
        neighbors.discard(crd_id)
        merged_values.append(", ".join(sorted(neighbors)) if neighbors else pd.NA)

    p["compared_to"] = pd.Series(
        pd.array(merged_values, dtype=DTYPE_STR), index=p.index
    )
    return p.drop(columns=[new_column])


def _join_unique_neighbors(values: pd.Series) -> str:
    """Return a deterministic comma-separated set for one grouped CRD_ID."""
    return ", ".join(sorted(set(values.dropna().astype(str))))


def _aggregate_distributed_neighbors(
    pairs_ddf,
    *,
    id_column: str,
    neighbor_column: str,
    staging_path: str,
    symmetric: bool = False,
):
    """Aggregate and stage a reusable distributed neighbor table."""
    directional = pairs_ddf[[id_column, neighbor_column]].rename(
        columns={id_column: "CRD_ID", neighbor_column: "_neighbor"}
    )
    if symmetric:
        reverse = pairs_ddf[[id_column, neighbor_column]].rename(
            columns={neighbor_column: "CRD_ID", id_column: "_neighbor"}
        )
        directional = dd.concat([directional, reverse])
    directional = directional.astype(
        {"CRD_ID": DTYPE_STR, "_neighbor": DTYPE_STR}
    )
    directional = directional[
        directional["CRD_ID"] != directional["_neighbor"]
    ]
    neighbor_meta = pd.Series(
        name="_new_compared_to",
        dtype="string",
        index=pd.Index([], name="CRD_ID", dtype="string"),
    )
    neighbors = (
        directional.groupby("CRD_ID")["_neighbor"]
        .apply(_join_unique_neighbors, meta=neighbor_meta)
        .to_frame()
        .reset_index()
    )
    _safe_to_parquet(neighbors, staging_path, write_index=False)
    return dd.read_parquet(staging_path, engine="pyarrow")


def _merge_distributed_neighbors(ddf, neighbors):
    """Merge one dataframe with a pre-aggregated neighbor table."""
    normalized_ddf = ddf.assign(CRD_ID=ddf["CRD_ID"].astype(DTYPE_STR))
    neighbors = neighbors.assign(CRD_ID=neighbors["CRD_ID"].astype(DTYPE_STR))
    merged = normalized_ddf.merge(neighbors, how="left", on="CRD_ID")
    meta = _merge_compared_to_column_partition(merged._meta)
    return merged.map_partitions(
        _merge_compared_to_column_partition,
        meta=meta,
    )


def _attach_distributed_neighbors(
    ddf,
    pairs_ddf,
    *,
    id_column: str,
    neighbor_column: str,
    staging_path: str,
    symmetric: bool = False,
):
    """Aggregate pairs once and attach their neighbors to one dataframe."""
    neighbors = _aggregate_distributed_neighbors(
        pairs_ddf,
        id_column=id_column,
        neighbor_column=neighbor_column,
        staging_path=staging_path,
        symmetric=symmetric,
    )
    return _merge_distributed_neighbors(ddf, neighbors)


def _ensure_compared_to_meta(meta_df: pd.DataFrame) -> pd.DataFrame:
    """Return a meta dataframe with a typed `compared_to` column.

    Args:
        meta_df: Input meta dataframe.

    Returns:
        pd.DataFrame: Meta dataframe with `compared_to` set to string dtype.
    """
    meta = meta_df.copy()
    meta["compared_to"] = pd.Series(pd.array([], dtype=DTYPE_STR))
    return meta


def _ensure_compared_to_column(part: pd.DataFrame) -> pd.DataFrame:
    """Ensure the partition has a `compared_to` column with string dtype.

    Args:
        part: Partition to update.

    Returns:
        pd.DataFrame: Partition with `compared_to` added when missing.
    """
    if "compared_to" not in part.columns:
        part = part.copy()
        part["compared_to"] = pd.Series(
            pd.array([pd.NA] * len(part), dtype=DTYPE_STR),
            index=part.index,
        )
    return part


def _get_expr_schema_hints(translation_config: dict | None) -> dict:
    """Return expr column schema hints from translation_config.

    Args:
        translation_config: Optional configuration containing expr hints.

    Returns:
        dict: Expr column schema hints or empty dict.
    """
    cfg = translation_config or {}
    hints = {}
    if cfg.get("save_expr_columns") is not False:
        hints.update(cfg.get("expr_column_schema", {}) or {})
    hints.update(cfg.get("runtime_schema_hints", {}) or {})
    return hints


def _ensure_compared_to(cat):
    """Ensure `compared_to` exists on a catalog for map_partitions.

    Args:
        cat: Catalog with a Dask backing dataframe.

    Returns:
        Catalog-like object: Catalog with `compared_to` ensured.
    """
    meta = _ensure_compared_to_meta(cat._ddf._meta)
    return cat.map_partitions(_ensure_compared_to_column, meta=meta)


# -----------------------
# Dtype normalization helpers (catalog-level, via map_partitions)
# -----------------------
_EXPECTED_TYPES = {
    "CRD_ID": DTYPE_STR,
    "id": DTYPE_STR,
    "ra": DTYPE_FLOAT,
    "dec": DTYPE_FLOAT,
    "z": DTYPE_FLOAT,
    "z_flag": DTYPE_FLOAT,
    "z_err": DTYPE_FLOAT,
    "instrument_type": DTYPE_STR,
    "survey": DTYPE_STR,
    "source": DTYPE_STR,
    "tie_result": DTYPE_INT8,
    "z_flag_homogenized": DTYPE_FLOAT,
    "instrument_type_homogenized": DTYPE_STR,
    "object_type_homogenized": DTYPE_STR,
    "compared_to": DTYPE_STR,
}


def _cast_partition_expected(
    part: pd.DataFrame,
    expected_types: dict,
    schema_hints: dict | None,
) -> pd.DataFrame:
    """Cast/add core columns and optional expr columns on a pandas partition.

    Args:
        part: Partition to cast.
        expected_types: Mapping of column name to expected dtype.
        schema_hints: Optional mapping of extra columns to type hints.

    Returns:
        pd.DataFrame: Partition with normalized dtypes.
    """
    df = part.copy()

    # 1) Core expected types
    for col, dtype in expected_types.items():
        if col not in df.columns:
            # Create missing column with target dtype
            if dtype is DTYPE_STR:
                df[col] = pd.Series(
                    pd.array([pd.NA] * len(df), dtype=DTYPE_STR), index=df.index
                )
            elif dtype is DTYPE_FLOAT:
                df[col] = pd.Series(
                    pd.array([np.nan] * len(df), dtype=DTYPE_FLOAT), index=df.index
                )
            elif dtype is DTYPE_INT8:
                df[col] = pd.Series(
                    pd.array([pd.NA] * len(df), dtype=DTYPE_INT8), index=df.index
                )
            elif dtype is DTYPE_INT:
                df[col] = pd.Series(
                    pd.array([pd.NA] * len(df), dtype=DTYPE_INT), index=df.index
                )
            elif dtype is DTYPE_BOOL:
                df[col] = pd.Series(
                    pd.array([pd.NA] * len(df), dtype=DTYPE_BOOL), index=df.index
                )
            continue

        # Cast existing column
        try:
            if dtype is DTYPE_STR:
                df[col] = _normalize_string_series_to_na(df[col])
            elif dtype is DTYPE_FLOAT:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(DTYPE_FLOAT)
            elif dtype is DTYPE_INT8:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(DTYPE_INT8)
            elif dtype is DTYPE_INT:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(DTYPE_INT)
            elif dtype is DTYPE_BOOL:
                df[col] = df[col].astype(DTYPE_BOOL)
        except Exception:
            # Be lenient: if cast fails, leave column as-is.
            pass

    # 2) Optional expr columns guided by hints (if provided)
    hints = dict(schema_hints or {})
    for col, kind in hints.items():
        k = str(kind).lower()
        if col not in df.columns:
            # Create missing column with target dtype
            try:
                if k == "str":
                    df[col] = pd.Series(
                        pd.array([pd.NA] * len(df), dtype=DTYPE_STR), index=df.index
                    )
                elif k == "float":
                    df[col] = pd.Series(
                        pd.array([np.nan] * len(df), dtype=DTYPE_FLOAT), index=df.index
                    )
                elif k == "int":
                    df[col] = pd.Series(
                        pd.array([pd.NA] * len(df), dtype=DTYPE_INT), index=df.index
                    )
                elif k == "bool":
                    df[col] = pd.Series(
                        pd.array([pd.NA] * len(df), dtype=DTYPE_BOOL), index=df.index
                    )
            except Exception:
                pass
            continue

        # Cast existing
        try:
            if k == "str":
                df[col] = _normalize_string_series_to_na(df[col])
            elif k == "float":
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(DTYPE_FLOAT)
            elif k == "int":
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(DTYPE_INT)
            elif k == "bool":
                df[col] = df[col].astype(DTYPE_BOOL)
        except Exception:
            pass

    return df


def _normalize_catalog_dtypes(cat, translation_config: dict | None):
    """Return a new catalog with normalized dtypes (lazy, via map_partitions).

    Args:
        cat: LSDB catalog with a Dask backing dataframe.
        translation_config: Optional config containing expr column hints.

    Returns:
        Catalog-like object: Catalog with normalized dtypes.
    """
    schema_hints_local = _get_expr_schema_hints(translation_config)

    # Build meta by running the caster on the meta dataframe
    meta_in = cat._ddf._meta
    meta_out = _cast_partition_expected(meta_in, _EXPECTED_TYPES, schema_hints_local)
    return cat.map_partitions(
        _cast_partition_expected, _EXPECTED_TYPES, schema_hints_local, meta=meta_out
    )


# -----------------------
# Dask/DDFrame normalization helpers (for the legacy path)
# -----------------------
def _coerce_optional_columns_for_import(
    df: dd.DataFrame,
    schema_hints: dict | None = None,
) -> dd.DataFrame:
    """Coerce expression columns to consistent Arrow dtypes across partitions.

    Args:
        df: Dask DataFrame to normalize.
        schema_hints: Optional mapping of expr column name to type.

    Returns:
        dd.DataFrame: Updated Dask DataFrame.
    """
    # Expression columns guided by hints
    hints = dict(schema_hints or {})
    if not hints:
        return df

    for col, kind in hints.items():
        k = str(kind).lower()
        if col not in df.columns:
            # Create missing with target dtype (avoid null[pyarrow] metas)
            if k == "str":
                df = _add_missing_with_dtype(df, col, DTYPE_STR)
            elif k == "float":
                df = _add_missing_with_dtype(df, col, DTYPE_FLOAT)
            elif k == "int":
                df = _add_missing_with_dtype(df, col, DTYPE_INT)
            elif k == "bool":
                df = _add_missing_with_dtype(df, col, DTYPE_BOOL)
            continue

        # Cast existing
        if k == "str":
            df[col] = df[col].map_partitions(
                _normalize_string_series_to_na,
                meta=pd.Series(pd.array([], dtype=DTYPE_STR)),
            )
        elif k == "float":
            coerced = dd.to_numeric(df[col], errors="coerce")
            df[col] = coerced.map_partitions(
                lambda s: s.astype(DTYPE_FLOAT),
                meta=pd.Series(pd.array([], dtype=DTYPE_FLOAT)),
            )
        elif k == "int":
            coerced = dd.to_numeric(df[col], errors="coerce")
            df[col] = coerced.map_partitions(
                lambda s: s.astype(DTYPE_INT),
                meta=pd.Series(pd.array([], dtype=DTYPE_INT)),
            )
        elif k == "bool":
            df[col] = df[col].map_partitions(
                lambda s: s.astype(DTYPE_BOOL),
                meta=pd.Series(pd.array([], dtype=DTYPE_BOOL)),
            )

    return df


def _normalize_ddf_expected_types(
    merged: dd.DataFrame, translation_config: dict | None
) -> dd.DataFrame:
    """Normalize expected/core columns on a merged Dask DataFrame prior to import.

    Args:
        merged: Merged Dask DataFrame.
        translation_config: Optional config containing expr column hints.

    Returns:
        dd.DataFrame: Updated Dask DataFrame.
    """
    expected_types = dict(_EXPECTED_TYPES)

    # 1) Core expected types
    for col, dtype in expected_types.items():
        if col not in merged.columns:
            continue
        try:
            if dtype == DTYPE_STR:
                merged[col] = merged[col].map_partitions(
                    _normalize_string_series_to_na,
                    meta=pd.Series(pd.array([], dtype=DTYPE_STR)),
                )
            elif dtype is DTYPE_FLOAT:
                merged[col] = dd.to_numeric(
                    merged[col], errors="coerce"
                ).map_partitions(
                    lambda s: s.astype(DTYPE_FLOAT),
                    meta=pd.Series(pd.array([], dtype=DTYPE_FLOAT)),
                )
            elif dtype is DTYPE_INT8:
                merged[col] = dd.to_numeric(
                    merged[col], errors="coerce"
                ).map_partitions(
                    lambda s: s.astype(DTYPE_INT8),
                    meta=pd.Series(pd.array([], dtype=DTYPE_INT8)),
                )
            elif dtype is DTYPE_INT:
                merged[col] = dd.to_numeric(
                    merged[col], errors="coerce"
                ).map_partitions(
                    lambda s: s.astype(DTYPE_INT),
                    meta=pd.Series(pd.array([], dtype=DTYPE_INT)),
                )
            elif dtype is DTYPE_BOOL:
                merged[col] = merged[col].map_partitions(
                    lambda s: s.astype(DTYPE_BOOL),
                    meta=pd.Series(pd.array([], dtype=DTYPE_BOOL)),
                )
        except Exception:
            # Best-effort only
            pass

    # 2) Prev/expr columns
    schema_hints_local = _get_expr_schema_hints(translation_config)
    merged = _coerce_optional_columns_for_import(merged, schema_hints_local)

    return merged


# -----------------------
# Parquet writer helper (legacy path)
# -----------------------
def _safe_to_parquet(ddf, path, **kwargs) -> None:
    """Write Parquet robustly for both plain Dask and nested_dask frames.

    Args:
        ddf: Dask DataFrame (or compatible) to write.
        path: Output path for Parquet dataset.
        **kwargs: Extra options forwarded to `to_parquet`.

    Raises:
        TypeError: If `to_parquet` fails for unexpected reasons.
    """
    try:
        ddf.to_parquet(path, engine="pyarrow", **kwargs)
    except TypeError as e:
        if "multiple values for keyword argument 'engine'" in str(e):
            ddf.to_parquet(path, **kwargs)
        else:
            raise


def _concat_and_write_hats(
    left_cat,
    right_cat,
    temp_dir: str,
    step,
    translation_config: dict | None,
    *,
    logger: logging.LoggerAdapter | None = None,
    log_steps: bool = False,
) -> str:
    """Normalize, concatenate catalogs, and write a HATS collection.

    Args:
        left_cat: Left-side catalog.
        right_cat: Right-side catalog.
        temp_dir: Output directory.
        step: Pipeline step identifier.
        translation_config: Optional configuration with schema hints.
        logger: Optional logger for progress messages.
        log_steps: Whether to log intermediate steps and durations.

    Returns:
        str: Output collection path.
    """
    if log_steps:
        t0 = time.time()
    left_fixed = _normalize_catalog_dtypes(left_cat, translation_config)
    right_fixed = _normalize_catalog_dtypes(right_cat, translation_config)
    if log_steps and logger is not None:
        logger.info(
            "Per-catalog type normalization attached (lazy) (%.2fs)",
            time.time() - t0,
        )

    if log_steps:
        t0 = time.time()
    merged_cat = left_fixed.concat(
        right_fixed,
        ignore_empty_margins=True,
    )
    if log_steps and logger is not None:
        logger.info("LSDB concat done (%.2fs)", time.time() - t0)

    if log_steps:
        t0 = time.time()
    collection_path = os.path.join(temp_dir, f"merged_step{step}_hats")
    merged_cat.write_catalog(
        collection_path,
        as_collection=True,
        overwrite=True,
    )
    if log_steps and logger is not None:
        logger.info(
            "Write complete (write_catalog): step=%s path=%s (%.2fs)",
            step,
            collection_path,
            time.time() - t0,
        )
    return collection_path


def _concat_parquet_import(
    left_cat,
    right_cat,
    temp_dir: str,
    logs_dir: str,
    step,
    client,
    translation_config: dict | None,
    *,
    logger: logging.LoggerAdapter | None = None,
    log_steps: bool = False,
) -> str:
    """Concatenate catalogs via Dask, write Parquet, then import a collection.

    Args:
        left_cat: Left-side catalog.
        right_cat: Right-side catalog.
        temp_dir: Output directory for Parquet.
        logs_dir: Path for import logs.
        step: Pipeline step identifier.
        client: Dask client used by the importer.
        translation_config: Optional configuration with schema hints.
        logger: Optional logger for progress messages.
        log_steps: Whether to log intermediate steps and durations.

    Returns:
        str: Output collection path.
    """
    if log_steps:
        t0 = time.time()
    lddf = left_cat._ddf
    rddf = right_cat._ddf
    merged = dd.concat([lddf, rddf])
    merged = _normalize_ddf_expected_types(merged, translation_config)
    if log_steps and logger is not None:
        logger.info(
            "Dask concat + type normalization (lazy) (%.2fs)",
            time.time() - t0,
        )

    if log_steps:
        t0 = time.time()
    merged_path = os.path.join(temp_dir, f"merged_step{step}")
    _safe_to_parquet(merged, merged_path, write_index=False)
    if log_steps and logger is not None:
        logger.info("Parquet written: path=%s (%.2fs)", merged_path, time.time() - t0)

    if log_steps:
        t0 = time.time()
    schema_hints_local = _get_expr_schema_hints(translation_config)
    schema_hints = schema_hints_local if schema_hints_local else None

    if log_steps and logger is not None:
        logger.info("START import_collection: step=%s parquet=%s", step, merged_path)
    collection_path = _build_collection_with_retry(
        parquet_path=merged_path,
        logs_dir=logs_dir,
        logger=logger,
        client=client,
        try_margin=True,
        schema_hints=schema_hints,
        margin_threshold=float(
            (translation_config or {}).get("margin_threshold_arcsec", 5.0)
        ),
    )
    if log_steps and logger is not None:
        logger.info(
            "END import_collection: step=%s path=%s (%.2fs)",
            step,
            collection_path,
            time.time() - t0,
        )
    return collection_path


def _distributed_pairs_update_and_import(
    left_cat,
    right_cat,
    pairs_ddf,
    temp_dir: str,
    logs_dir: str,
    step,
    client,
    translation_config: dict | None,
    logger: logging.LoggerAdapter,
) -> str:
    """Update both catalogs without embedding global adjacency in task graphs."""
    staging_root = os.path.join(temp_dir, f"distributed_pairs_step{step}")
    merged_path = os.path.join(staging_root, "merged")
    collection_path_target = os.path.join(temp_dir, f"merged_step{step}_hats")
    logger.info(
        "Distributed compared_to update: step=%s staging=%s",
        step,
        staging_root,
    )

    neighbors = _aggregate_distributed_neighbors(
        pairs_ddf,
        id_column="CRD_IDleft",
        neighbor_column="CRD_IDright",
        staging_path=os.path.join(staging_root, "neighbors"),
        symmetric=True,
    )
    left_ddf = _merge_distributed_neighbors(left_cat._ddf, neighbors)
    right_ddf = _merge_distributed_neighbors(right_cat._ddf, neighbors)
    merged = dd.concat([left_ddf, right_ddf])
    merged = _normalize_ddf_expected_types(merged, translation_config)
    _safe_to_parquet(merged, merged_path, write_index=False)
    logger.info("Distributed updated Parquet written: step=%s path=%s", step, merged_path)

    schema_hints_local = _get_expr_schema_hints(translation_config)
    collection_path = _build_collection_with_retry(
        parquet_path=merged_path,
        logs_dir=logs_dir,
        logger=logger,
        client=client,
        try_margin=True,
        schema_hints=schema_hints_local or None,
        margin_threshold=float(
            (translation_config or {}).get("margin_threshold_arcsec", 5.0)
        ),
        output_path=collection_path_target,
        output_artifact_name=f"merged_step{step}_hats",
        catalog_artifact_name=f"merged_step{step}",
    )
    shutil.rmtree(staging_root, ignore_errors=True)
    return collection_path


# -----------------------
# Main logic
# -----------------------
def crossmatch_tiebreak(
    left_cat,
    right_cat,
    logs_dir: str,
    temp_dir: str,
    step,
    client,  # kept for signature compatibility (used in legacy import path)
    translation_config: dict | None = None,
    do_import: bool = True,  # kept for signature compatibility; ignored in both paths (we always return a collection)
) -> str:
    """Crossmatch two catalogs, update `compared_to`, then export a collection.

    Args:
        left_cat: Left-side catalog.
        right_cat: Right-side catalog.
        logs_dir: Path for import logs (legacy path).
        temp_dir: Path for temporary output (Parquet or HATS).
        step: Pipeline step identifier.
        client: Dask client (legacy import path).
        translation_config: Optional configuration with crossmatch and schema hints.
        do_import: Ignored; kept for signature compatibility.

    Returns:
        str: Output collection path.
    """
    logger = _get_logger()
    t0_all = time.time()

    # Parameters with defaults
    radius = float((translation_config or {}).get("crossmatch_radius_arcsec", 0.5))
    k = int((translation_config or {}).get("crossmatch_n_neighbors", 160))
    saturation_enabled = bool(
        (translation_config or {}).get("crossmatch_saturation_enabled", False)
    )
    geometry_diagnostics_enabled = bool(
        (translation_config or {}).get("crossmatch_geometry_diagnostics_enabled", False)
    )
    warn_fraction = 0.01
    fail_raw = None
    if saturation_enabled:
        warn_fraction = float(
            (translation_config or {}).get("crossmatch_saturation_warn_fraction", 0.01)
        )
        fail_raw = (translation_config or {}).get(
            "crossmatch_saturation_fail_fraction", 0.10
        )
    fail_fraction = None if fail_raw is None else float(fail_raw)
    if saturation_enabled and not 0.0 <= warn_fraction <= 1.0:
        raise ValueError("crossmatch_saturation_warn_fraction must be in [0, 1]")
    if (
        saturation_enabled
        and fail_fraction is not None
        and not warn_fraction <= fail_fraction <= 1.0
    ):
        raise ValueError(
            "crossmatch_saturation_fail_fraction must be null or in "
            "[crossmatch_saturation_warn_fraction, 1]"
        )
    total_by_source = _catalog_source_totals(left_cat) if saturation_enabled else {}

    logger.info(
        'START crossmatch_update_compared_to: step=%s radius=%.3f" n_neighbors=%d backend=%s',
        step,
        radius,
        k,
        _get_backend_label(),
    )

    # 1) Spatial crossmatch
    t0 = time.time()
    pair_left = project_catalog_for_pair_crossmatch(
        left_cat,
        include_source=saturation_enabled,
    )
    pair_right = project_catalog_for_pair_crossmatch(
        right_cat,
        include_source=False,
    )
    xmatched = pair_left.crossmatch(
        pair_right,
        radius_arcsec=radius,
        n_neighbors=k,
        suffixes=("left", "right"),
        suffix_method="all_columns",
    )
    logger.info("Crossmatch done (%.2fs)", time.time() - t0)

    # 2) Project and stage pair columns inside workers. No pair rows are
    # gathered on the driver in the production path.
    t0 = time.time()
    pair_cols = ["CRD_IDleft", "CRD_IDright"]
    if geometry_diagnostics_enabled and "_dist_arcsec" in xmatched._ddf.columns:
        pair_cols.append("_dist_arcsec")
    if saturation_enabled and "sourceleft" in xmatched._ddf.columns:
        pair_cols.append("sourceleft")
    staging_root = os.path.join(temp_dir, f"distributed_pairs_step{step}")
    raw_pairs_path = os.path.join(staging_root, "raw_pairs")
    pairs_path = os.path.join(staging_root, "pairs")
    shutil.rmtree(staging_root, ignore_errors=True)
    os.makedirs(staging_root, exist_ok=True)
    raw_pairs = stage_projected_pairs(xmatched._ddf, pair_cols, raw_pairs_path)
    pairs = raw_pairs.astype(
        {"CRD_IDleft": DTYPE_STR, "CRD_IDright": DTYPE_STR}
    )
    pairs = pairs[pairs["CRD_IDleft"] != pairs["CRD_IDright"]]
    pairs = pairs.drop_duplicates(subset=["CRD_IDleft", "CRD_IDright"])
    _safe_to_parquet(pairs, pairs_path, write_index=False)
    pairs = dd.read_parquet(pairs_path, engine="pyarrow")

    pair_count, node_count = dask.compute(
        pairs.map_partitions(len).sum(),
        dd.concat([pairs["CRD_IDleft"], pairs["CRD_IDright"]]).nunique(),
    )
    pair_count = int(pair_count)
    node_count = int(node_count)
    total_links = 2 * pair_count
    if not pair_count:
        logger.info("No pairs found; `compared_to` remains unchanged.")
    if saturation_enabled and pair_count:
        _log_neighbor_saturation_distributed(
            pairs,
            id_col="CRD_IDleft",
            source_col="sourceleft" if "sourceleft" in pairs.columns else None,
            limit=k,
            logger=logger,
            context=f"crossmatch step={step}",
            total_by_source=total_by_source,
            warn_fraction=warn_fraction,
            fail_fraction=fail_fraction,
        )
    if geometry_diagnostics_enabled and pair_count:
        diagnostic_pairs = pairs.compute()
        log_pair_separation_diagnostics(
            diagnostic_pairs,
            left_col="CRD_IDleft",
            right_col="CRD_IDright",
            radius_arcsec=radius,
            logger=logger,
            context=f"crossmatch step={step}",
        )
        diagnostic_adj = _adjacency_from_pairs(
            diagnostic_pairs["CRD_IDleft"], diagnostic_pairs["CRD_IDright"]
        )
        log_component_size_diagnostics(
            diagnostic_adj,
            logger=logger,
            context=f"crossmatch step={step}",
        )
    logger.info(
        "Pair summary built without global adjacency: links=%d nodes=%d (%.2fs)",
        total_links,
        node_count,
        time.time() - t0,
    )

    # With no new edges, retain the cheaper spatial concat path.
    if total_links == 0:
        collection_path = _concat_and_write_hats(
            left_cat,
            right_cat,
            temp_dir,
            step,
            translation_config,
            logger=logger,
            log_steps=True,
        )
        logger.info(
            "END crossmatch_update_compared_to: step=%s links=%d nodes=%d output=%s (%.2fs)",
            step,
            total_links,
            node_count,
            collection_path,
            time.time() - t0_all,
        )
        shutil.rmtree(staging_root, ignore_errors=True)
        return collection_path

    # Aggregate and join neighbor strings in the distributed dataframe. The
    # resulting Parquet is reimported so spatial HATS metadata and margins are
    # rebuilt from the updated rows.
    collection_path = _distributed_pairs_update_and_import(
        left_cat,
        right_cat,
        pairs,
        temp_dir,
        logs_dir,
        step,
        client,
        translation_config,
        logger,
    )
    logger.info(
        "END crossmatch_update_compared_to: step=%s links=%d nodes=%d output=%s (%.2fs)",
        step,
        total_links,
        node_count,
        collection_path,
        time.time() - t0_all,
    )
    return collection_path


def crossmatch_tiebreak_safe(
    left_cat,
    right_cat,
    logs_dir: str,
    temp_dir: str,
    step,
    client,  # used in legacy import path
    translation_config: dict | None = None,
    do_import: bool = True,  # ignored; always returns a collection
) -> str:
    """Wrap `crossmatch_tiebreak` with a graceful empty-overlap fallback.

    If the crossmatch yields a known empty-overlap condition:
        - LSDB path: ensure `compared_to`, normalize each catalog, LSDB concat,
          `write_catalog`, return collection.
        - Legacy path: ensure `compared_to`, Dask concat, Parquet, import,
          return collection.

    Args:
        left_cat: Left-side catalog.
        right_cat: Right-side catalog.
        logs_dir: Path for import logs (legacy path).
        temp_dir: Path for temporary output (Parquet or HATS).
        step: Pipeline step identifier.
        client: Dask client (legacy import path).
        translation_config: Optional configuration with schema hints.
        do_import: Ignored; kept for signature compatibility.

    Returns:
        str: Output collection path.

    Raises:
        RuntimeError: Re-raised if not an empty-overlap condition.
    """
    logger = _get_logger()
    t0_safe = time.time()
    logger.info(
        "START xmatch_update_compared_to_safe: step=%s backend=%s",
        step,
        _get_backend_label(),
    )

    try:
        out = crossmatch_tiebreak(
            left_cat=left_cat,
            right_cat=right_cat,
            logs_dir=logs_dir,
            temp_dir=temp_dir,
            step=step,
            client=client,
            translation_config=translation_config,
            do_import=do_import,  # ignored internally
        )
        logger.info(
            "END xmatch_update_compared_to_safe: step=%s output=%s (%.2fs)",
            step,
            out,
            time.time() - t0_safe,
        )
        return out

    except RuntimeError as e:
        msg = str(e)
        if ("The output catalog is empty" in msg) or ("Catalogs do not overlap" in msg):
            logger.info("Empty-overlap condition detected: %s", msg)

            # Ensure `compared_to` exists on both sides
            left_ready = _ensure_compared_to(left_cat)
            right_ready = _ensure_compared_to(right_cat)

            if USE_LSDB_CONCAT:
                collection_path = _concat_and_write_hats(
                    left_ready,
                    right_ready,
                    temp_dir,
                    step,
                    translation_config,
                )
                logger.info(
                    "END xmatch_update_compared_to_safe: step=%s output=%s (%.2fs)",
                    step,
                    collection_path,
                    time.time() - t0_safe,
                )
                return collection_path

            collection_path = _concat_parquet_import(
                left_ready,
                right_ready,
                temp_dir,
                logs_dir,
                step,
                client,
                translation_config,
                logger=logger,
            )
            logger.info(
                "END xmatch_update_compared_to_safe: step=%s output=%s (%.2fs)",
                step,
                collection_path,
                time.time() - t0_safe,
            )
            return collection_path

        # Unexpected exceptions are re-raised
        raise
