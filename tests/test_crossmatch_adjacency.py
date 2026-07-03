import sys
import types
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages"))
if "tables_io" not in sys.modules:
    tables_io = types.ModuleType("tables_io")
    tables_io.types = types.SimpleNamespace(PD_DATAFRAME="PD_DATAFRAME")
    sys.modules["tables_io"] = tables_io

from crossmatch_auto import (  # noqa: E402
    _adjacency_from_pairs as auto_adjacency,
    _merge_compared_to_partition as auto_merge_compared_to,
)
from crossmatch_cross import (  # noqa: E402
    _adjacency_from_pairs as cross_adjacency,
    _merge_compared_to_partition as cross_merge_compared_to,
)


@pytest.mark.parametrize("merge", [auto_merge_compared_to, cross_merge_compared_to])
def test_compared_to_merge_is_normalized_and_idempotent(merge):
    frame = pd.DataFrame(
        {
            "CRD_ID": ["A", "B", "C"],
            "compared_to": ["B, B, A", ["A", "C"], pd.NA],
        }
    )
    adjacency = {"A": {"C", "A"}, "B": {"A"}, "C": {"B"}}

    once = merge(frame, adjacency)
    twice = merge(once, adjacency)

    assert once["compared_to"].astype("string").tolist() == ["B, C", "A, C", "B"]
    pd.testing.assert_series_equal(once["compared_to"], twice["compared_to"])


@pytest.mark.parametrize("adjacency", [auto_adjacency, cross_adjacency])
def test_adjacency_is_symmetric_deduplicated_and_excludes_self(adjacency):
    left = pd.Series(["A", "A", "A", "B"])
    right = pd.Series(["B", "B", "A", "C"])

    result = adjacency(left, right)

    assert result == {"A": {"B"}, "B": {"A", "C"}, "C": {"B"}}
