"""Regression tests for strict crossmatch schema normalization."""

import sys
import types
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages"))
if "tables_io" not in sys.modules:
    tables_io = types.ModuleType("tables_io")
    tables_io.types = types.SimpleNamespace(PD_DATAFRAME="PD_DATAFRAME")
    sys.modules["tables_io"] = tables_io

from crossmatch_cross import (  # noqa: E402
    _EXPECTED_TYPES,
    _cast_partition_expected,
    _get_expr_schema_hints,
)
from specz import DTYPE_STR, build_runtime_schema_hints  # noqa: E402


def test_object_type_string_backends_normalize_to_same_dtype():
    python_string = pd.DataFrame(
        {"object_type_homogenized": pd.Series(["star"], dtype="string")}
    )
    pyarrow_string = pd.DataFrame(
        {"object_type_homogenized": pd.Series(["agn"], dtype="string[pyarrow]")}
    )

    left = _cast_partition_expected(python_string, _EXPECTED_TYPES, None)
    right = _cast_partition_expected(pyarrow_string, _EXPECTED_TYPES, None)

    assert left["object_type_homogenized"].dtype == DTYPE_STR
    assert right["object_type_homogenized"].dtype == DTYPE_STR
    assert left["object_type_homogenized"].dtype == right["object_type_homogenized"].dtype


def test_runtime_schema_combines_expr_extra_priority_and_enabled_flags():
    param_config = {
        "extra_columns": {
            "label": "str",
            "external_score": {"source": "RAW_SCORE", "type": "float"},
        },
        "insert_DP1_footprint_flag": True,
        "insert_rubin_footprint_flag": False,
    }
    translation_config = {
        "save_expr_columns": True,
        "expr_column_schema": {"expr_value": "int", "label": "float"},
        "tiebreaking_priority": ["z_flag_homogenized", "custom_rank"],
    }

    hints = build_runtime_schema_hints(param_config, translation_config)

    assert hints == {
        "expr_value": "int",
        "label": "str",
        "external_score": "float",
        "is_in_DP1_fields": "int",
        "custom_rank": "float",
    }


def test_crossmatch_runtime_hints_override_expr_hints_and_normalize_extra_columns():
    config = {
        "save_expr_columns": True,
        "expr_column_schema": {"label": "float", "expr_value": "int"},
        "runtime_schema_hints": {"label": "str", "extra_text": "str"},
    }
    hints = _get_expr_schema_hints(config)
    assert hints == {"label": "str", "expr_value": "int", "extra_text": "str"}

    left = pd.DataFrame(
        {
            "label": pd.Series(["left"], dtype="string"),
            "extra_text": pd.Series(["a"], dtype="string"),
        }
    )
    right = pd.DataFrame(
        {
            "label": pd.Series(["right"], dtype="string[pyarrow]"),
            "extra_text": pd.Series(["b"], dtype="string[pyarrow]"),
        }
    )
    left_fixed = _cast_partition_expected(left, _EXPECTED_TYPES, hints)
    right_fixed = _cast_partition_expected(right, _EXPECTED_TYPES, hints)

    assert left_fixed["label"].dtype == right_fixed["label"].dtype == DTYPE_STR
    assert left_fixed["extra_text"].dtype == right_fixed["extra_text"].dtype == DTYPE_STR
