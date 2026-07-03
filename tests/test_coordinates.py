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

from specz import _validate_ra_dec_or_fail  # noqa: E402


def _frame(ra, dec):
    return dd.from_pandas(
        pd.DataFrame({"CRD_ID": ["A"], "ra": [ra], "dec": [dec]}),
        npartitions=1,
        sort=False,
    )


@pytest.mark.parametrize(
    ("ra", "dec"),
    [(0.0, -90.0), (359.999999, 90.0), (180.0, 0.0)],
)
def test_coordinate_validation_accepts_boundaries(ra, dec):
    _validate_ra_dec_or_fail(_frame(ra, dec), "demo")


@pytest.mark.parametrize(
    ("ra", "dec"),
    [
        (-0.001, 0.0),
        (360.0, 0.0),
        (0.0, -90.001),
        (0.0, 90.001),
        (float("nan"), 0.0),
        (0.0, float("inf")),
    ],
)
def test_coordinate_validation_rejects_invalid_values(ra, dec):
    with pytest.raises(ValueError, match="Invalid RA/DEC rows"):
        _validate_ra_dec_or_fail(_frame(ra, dec), "demo")
