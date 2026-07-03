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

from specz import _generate_crd_ids, _validate_unique_crd_ids  # noqa: E402


def test_generate_crd_ids_are_unique_across_partitions():
    frame = dd.from_pandas(
        pd.DataFrame({"value": range(17)}), npartitions=4, sort=False
    )

    result = _generate_crd_ids(frame, "314_example", "/tmp").compute()

    assert result["CRD_ID"].is_unique
    assert set(result["CRD_ID"]) == {f"CRD314_{i}" for i in range(1, 18)}


def test_validate_unique_crd_ids_rejects_collisions():
    frame = dd.from_pandas(
        pd.DataFrame({"CRD_ID": ["CRD314_1", "CRD314_1"]}),
        npartitions=2,
        sort=False,
    )

    with pytest.raises(RuntimeError, match="non-unique CRD_ID"):
        _validate_unique_crd_ids(frame, "314_example", logging.getLogger(__name__))


def test_generate_crd_ids_is_deterministic_for_same_partitioned_input():
    frame = dd.from_pandas(
        pd.DataFrame({"value": range(12)}), npartitions=3, sort=False
    )

    first = _generate_crd_ids(frame, "314_example", "/tmp").compute()
    second = _generate_crd_ids(frame, "314_example", "/tmp").compute()

    assert first[["value", "CRD_ID"]].to_dict("records") == second[
        ["value", "CRD_ID"]
    ].to_dict("records")
