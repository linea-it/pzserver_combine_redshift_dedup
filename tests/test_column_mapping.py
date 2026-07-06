import logging
import sys
import types
from pathlib import Path

import dask.dataframe as dd
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages"))
if "tables_io" not in sys.modules:
    tables_io = types.ModuleType("tables_io")
    tables_io.types = types.SimpleNamespace(PD_DATAFRAME="PD_DATAFRAME")
    sys.modules["tables_io"] = tables_io

from specz import _validate_and_rename  # noqa: E402

LOGGER = logging.getLogger("test.column_mapping")


def _ddf(data):
    return dd.from_pandas(pd.DataFrame(data), npartitions=1, sort=False)


def test_column_mapping_rejects_missing_configured_source():
    entry = {
        "internal_name": "1_demo",
        "columns": {"id": "missing_source"},
    }

    with pytest.raises(ValueError, match="missing_source"):
        _validate_and_rename(_ddf({"other": [1]}), entry, LOGGER)


def test_column_mapping_parks_existing_target_before_rename():
    frame = _ddf(
        {
            "raw_id": ["new"],
            "id": ["original"],
            "ra_raw": [10.0],
            "dec_raw": [-20.0],
        }
    )
    entry = {
        "internal_name": "1_demo",
        "columns": {"id": "raw_id", "ra": "ra_raw", "dec": "dec_raw"},
    }

    result = _validate_and_rename(frame, entry, LOGGER).compute()

    assert result.loc[0, "id"] == "new"
    assert result.loc[0, "id__orig"] == "original"
    assert result.loc[0, "ra"] == 10.0
    assert result.loc[0, "dec"] == -20.0
    assert result.loc[0, "source"] == "1_demo"


def test_column_mapping_ignores_null_mapping_and_creates_base_schema():
    entry = {
        "internal_name": "1_demo",
        "columns": {"z_err": None},
    }

    result = _validate_and_rename(_ddf({"unrelated": [1]}), entry, LOGGER)

    expected = {
        "id",
        "instrument_type",
        "survey",
        "ra",
        "dec",
        "z",
        "z_flag",
        "z_err",
        "source",
    }
    assert expected.issubset(result.columns)
    assert pd.isna(result.compute().loc[0, "z_err"])
