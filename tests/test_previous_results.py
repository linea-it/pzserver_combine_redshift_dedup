from __future__ import annotations

import logging
import sys
from pathlib import Path

import dask.dataframe as dd
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages"))

from specz import (  # noqa: E402
    _drop_previous_results,
    _copy_extra_columns_from_sources,
    _normalize_extra_columns_config,
    _select_output_columns,
    _validate_and_rename,
)


def test_drop_previous_results_removes_current_and_historical_columns():
    pdf = pd.DataFrame(
        {
            "id": ["source-1"],
            "CRD_ID": ["CRD001_1"],
            "compared_to": ["CRD002_1"],
            "group_id": [42],
            "CRD_ID_prev": ["CRD000_1"],
            "CRD_ID_prev2": ["CRD999_1"],
            "compared_to_prev3": ["CRD998_1"],
            "group_id_prev": [41],
            "ra": [10.0],
        }
    )

    result = _drop_previous_results(dd.from_pandas(pdf, npartitions=1), logging.getLogger())

    assert list(result.columns) == ["id", "ra"]


def test_crd_id_mapped_to_id_is_retained_before_cleanup():
    pdf = pd.DataFrame(
        {
            "CRD_ID": ["CRD011_162"],
            "compared_to": ["CRD012_9"],
            "group_id": [123],
            "CRD_ID_prev": ["CRD010_8"],
            "ra": [34.49],
            "dec": [-5.56],
            "z": [0.45],
        }
    )
    entry = {
        "internal_name": "019_pipeline_sample",
        "columns": {
            "id": "CRD_ID",
            "ra": "ra",
            "dec": "dec",
            "z": "z",
        },
    }
    frame = dd.from_pandas(pdf, npartitions=1)

    renamed = _validate_and_rename(frame, entry, logging.getLogger())
    result = _drop_previous_results(renamed, logging.getLogger()).compute()

    assert result.loc[0, "id"] == "CRD011_162"
    assert not any(
        col in {"CRD_ID", "compared_to", "group_id"}
        or col.startswith(("CRD_ID_prev", "compared_to_prev", "group_id_prev"))
        for col in result.columns
    )


def test_extra_column_is_preserved_and_cast():
    frame = dd.from_pandas(
        pd.DataFrame({"id": ["a", "b"], "DELTACHI2": ["12.5", "invalid"]}),
        npartitions=1,
    )

    extra_columns = _normalize_extra_columns_config({"DELTACHI2": "float"})
    result = _select_output_columns(
        frame,
        translation_rules_uc={},
        tiebreaking_priority=[],
        used_type_fastpath=False,
        extra_columns=extra_columns,
    ).compute()

    assert result["DELTACHI2"].tolist()[0] == 12.5
    assert pd.isna(result["DELTACHI2"].tolist()[1])


def test_missing_extra_column_is_created_with_null_values():
    frame = dd.from_pandas(pd.DataFrame({"id": ["a", "b"]}), npartitions=1)

    extra_columns = _normalize_extra_columns_config({"DELTACHI2": "float"})
    result = _select_output_columns(
        frame,
        translation_rules_uc={},
        tiebreaking_priority=[],
        used_type_fastpath=False,
        extra_columns=extra_columns,
    ).compute()

    assert result["DELTACHI2"].isna().all()
    assert str(result["DELTACHI2"].dtype) == "double[pyarrow]"


def test_extra_columns_rejects_unsupported_dtype():
    try:
        _normalize_extra_columns_config({"DELTACHI2": "decimal"})
    except ValueError as exc:
        assert "Supported types: str, float, int, bool" in str(exc)
    else:
        raise AssertionError("Unsupported extra column dtype was accepted")


def test_extra_column_source_is_copied_before_id_mapping_collision():
    pdf = pd.DataFrame(
        {
            "TARGETID": ["target-1"],
            "id": ["original-1"],
            "ra": [10.0],
            "dec": [-5.0],
            "z": [0.3],
        }
    )
    entry = {
        "internal_name": "020_collision",
        "columns": {
            "id": "TARGETID",
            "ra": "ra",
            "dec": "dec",
            "z": "z",
        },
    }
    extra_columns = _normalize_extra_columns_config(
        {"ORIGINAL_ID": {"source": "id", "type": "str"}}
    )
    frame = dd.from_pandas(pdf, npartitions=1)

    frame = _copy_extra_columns_from_sources(
        frame, extra_columns, logging.getLogger()
    )
    frame = _validate_and_rename(frame, entry, logging.getLogger())
    result = _select_output_columns(
        frame,
        translation_rules_uc={},
        tiebreaking_priority=[],
        used_type_fastpath=False,
        extra_columns=extra_columns,
    ).compute()

    assert result.loc[0, "id"] == "target-1"
    assert result.loc[0, "ORIGINAL_ID"] == "original-1"


def test_missing_configured_source_produces_null_output():
    frame = dd.from_pandas(
        pd.DataFrame({"ORIGINAL_ID": ["must-not-be-used"]}), npartitions=1
    )
    extra_columns = _normalize_extra_columns_config(
        {"ORIGINAL_ID": {"source": "id", "type": "str"}}
    )

    frame = _copy_extra_columns_from_sources(
        frame, extra_columns, logging.getLogger()
    )
    result = _select_output_columns(
        frame,
        translation_rules_uc={},
        tiebreaking_priority=[],
        used_type_fastpath=False,
        extra_columns=extra_columns,
    ).compute()

    assert pd.isna(result.loc[0, "ORIGINAL_ID"])
