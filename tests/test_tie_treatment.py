import sys
from pathlib import Path

import pandas as pd
import pytest
import dask.dataframe as dd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages"))

from deduplication import (  # noqa: E402
    filter_dask_by_tie_treatment,
    filter_pandas_by_tie_treatment,
)


def _results():
    return pd.DataFrame(
        {
            "CRD_ID": ["winner", "loser", "tie-a", "tie-b", "star"],
            "group_id": [1, 1, 2, 2, 3],
            "tie_result": [1, 0, 2, 2, 3],
        }
    )


@pytest.mark.parametrize(
    ("option", "expected"),
    [
        ("remove_all", {"winner"}),
        ("keep_all", {"winner", "tie-a", "tie-b"}),
        ("invalid", {"winner"}),
    ],
)
def test_final_tie_filtering_policies(option, expected):
    result, effective, resolved = filter_pandas_by_tie_treatment(_results(), option)

    assert set(result["CRD_ID"]) == expected
    assert "star" not in set(result["CRD_ID"])
    assert effective == (option if option != "invalid" else "remove_all")
    assert resolved == 0


def test_draw_one_keeps_exactly_one_candidate_per_hard_tie():
    result, effective, resolved = filter_pandas_by_tie_treatment(
        _results(), "draw_one", random_state=42
    )

    assert effective == "draw_one"
    assert resolved == 1
    assert set(result["CRD_ID"]) == {"winner", "tie-a"}
    assert "star" not in set(result["CRD_ID"])


def test_draw_one_without_group_id_falls_back_to_remove_all():
    frame = _results().drop(columns="group_id")

    result, effective, resolved = filter_pandas_by_tie_treatment(frame, "draw_one")

    assert set(result["CRD_ID"]) == {"winner"}
    assert effective == "remove_all"
    assert resolved == 0


def test_final_tie_filter_requires_tie_result():
    with pytest.raises(RuntimeError, match="tie_result"):
        filter_pandas_by_tie_treatment(pd.DataFrame({"CRD_ID": ["A"]}), "remove_all")


def test_distributed_draw_one_is_lazy_and_deterministic():
    frame = pd.DataFrame(
        {
            "CRD_ID": ["winner", "loser", "tie-b", "tie-a", "tie-d", "tie-c"],
            "group_id": [1, 1, 2, 2, 3, 3],
            "tie_result": [1, 0, 2, 2, 2, 2],
        }
    )
    distributed = dd.from_pandas(frame, npartitions=3, sort=False)

    result, effective = filter_dask_by_tie_treatment(distributed, "draw_one")

    assert isinstance(result, dd.DataFrame)
    assert effective == "draw_one"
    assert set(result.compute()["CRD_ID"]) == {"winner", "tie-a", "tie-c"}


def test_distributed_draw_one_without_group_id_falls_back_to_remove_all():
    distributed = dd.from_pandas(_results().drop(columns="group_id"), npartitions=2)

    result, effective = filter_dask_by_tie_treatment(distributed, "draw_one")

    assert effective == "remove_all"
    assert set(result.compute()["CRD_ID"]) == {"winner"}
