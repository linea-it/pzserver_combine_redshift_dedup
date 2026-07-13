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
        pd.DataFrame(
            {
                "ra": [float(i) for i in range(17)],
                "dec": [float(-i) for i in range(17)],
                "z": [float(i) / 10.0 for i in range(17)],
            }
        ),
        npartitions=4,
        sort=False,
    )

    result = _generate_crd_ids(frame, "314_example", "/tmp").compute()

    assert result["CRD_ID"].is_unique
    assert result["CRD_ID"].str.match(r"^CRD314_[0-9a-f]{16}$").all()


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
        pd.DataFrame(
            {
                "ra": [float(i) for i in range(12)],
                "dec": [float(i + 1) for i in range(12)],
                "z": [float(i + 2) for i in range(12)],
            }
        ),
        npartitions=3,
        sort=False,
    )

    first = _generate_crd_ids(frame, "314_example", "/tmp").compute()
    second = _generate_crd_ids(frame, "314_example", "/tmp").compute()

    assert first[["ra", "dec", "z", "CRD_ID"]].to_dict("records") == second[
        ["ra", "dec", "z", "CRD_ID"]
    ].to_dict("records")


def test_generate_crd_ids_do_not_depend_on_row_order_or_partitions():
    pdf = pd.DataFrame(
        {
            "id": ["duplicate", "duplicate", None, "ignored"],
            "ra": [10.0, 20.0, 30.0, 40.0],
            "dec": [-10.0, -20.0, -30.0, -40.0],
            "z": [0.1, 0.2, 0.3, 0.4],
        }
    )
    forward = dd.from_pandas(pdf, npartitions=2, sort=False)
    reverse = dd.from_pandas(
        pdf.iloc[::-1].reset_index(drop=True), npartitions=3, sort=False
    )

    first = _generate_crd_ids(forward, "314_example", "/tmp").compute()
    second = _generate_crd_ids(reverse, "314_example", "/tmp").compute()

    key = ["ra", "dec", "z"]
    first_map = first.set_index(key)["CRD_ID"].to_dict()
    second_map = second.set_index(key)["CRD_ID"].to_dict()
    assert first_map == second_map


def test_generate_crd_ids_duplicate_ra_dec_z_collide_for_validation():
    frame = dd.from_pandas(
        pd.DataFrame(
            {
                "ra": [10.0, 10.0],
                "dec": [-10.0, -10.0],
                "z": [0.1, 0.1],
            }
        ),
        npartitions=2,
        sort=False,
    )

    result = _generate_crd_ids(frame, "314_example", "/tmp")

    with pytest.raises(RuntimeError, match="non-unique CRD_ID"):
        _validate_unique_crd_ids(result, "314_example", logging.getLogger(__name__))
