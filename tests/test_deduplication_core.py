import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages"))

from deduplication import deduplicate_pandas  # noqa: E402


INSTRUMENT_PRIORITY = {"s": 3, "g": 2, "p": 1}


def _row(crd_id, compared_to, *, flag=4.0, instrument="s", z=0.1):
    return {
        "CRD_ID": crd_id,
        "compared_to": compared_to,
        "z": z,
        "z_flag_homogenized": flag,
        "instrument_type_homogenized": instrument,
    }


def _deduplicate(rows, *, threshold=0.001):
    return deduplicate_pandas(
        pd.DataFrame(rows),
        tiebreaking_priority=[
            "z_flag_homogenized",
            "instrument_type_homogenized",
        ],
        instrument_type_priority=INSTRUMENT_PRIORITY,
        delta_z_threshold=threshold,
        group_col="group_id",
    ).set_index("CRD_ID")


def test_full_dedup_respects_ordered_priorities():
    result = _deduplicate(
        [
            _row("A", "B, C", flag=3, instrument="s"),
            _row("B", "A, C", flag=4, instrument="g"),
            _row("C", "A, B", flag=4, instrument="s"),
        ]
    )

    assert result["tie_result"].astype(int).to_dict() == {"A": 0, "B": 0, "C": 1}
    assert result["group_id"].nunique() == 1


def test_full_dedup_emits_two_and_three_way_hard_ties():
    two_way = _deduplicate(
        [
            _row("A", "B", z=0.10),
            _row("B", "A", z=0.20),
        ]
    )
    three_way = _deduplicate(
        [
            _row("C", "D", z=0.10),
            _row("D", "C, E", z=0.20),
            _row("E", "D", z=0.30),
        ]
    )

    assert two_way["tie_result"].astype(int).tolist() == [2, 2]
    assert three_way["tie_result"].astype(int).tolist() == [2, 2, 2]
    assert two_way["group_id"].nunique() == 1
    assert three_way["group_id"].nunique() == 1


def test_stars_are_isolated_from_nonstar_deduplication():
    result = _deduplicate(
        [
            _row("A", "S", flag=4),
            _row("S", "A, B", flag=6),
            _row("B", "S", flag=3),
        ]
    )

    assert result["tie_result"].astype(int).to_dict() == {"A": 1, "S": 3, "B": 1}
    assert result["group_id"].nunique() == 3


def test_connected_components_are_transitive():
    result = _deduplicate(
        [
            _row("A", "B", flag=2),
            _row("B", "A, C", flag=3),
            _row("C", "B", flag=4),
        ]
    )

    assert result["group_id"].nunique() == 1
    assert result["tie_result"].astype(int).to_dict() == {"A": 0, "B": 0, "C": 1}


def test_empty_catalog_is_supported():
    frame = pd.DataFrame(
        {
            "CRD_ID": pd.Series(dtype="string"),
            "compared_to": pd.Series(dtype="string"),
            "z": pd.Series(dtype="float64"),
            "z_flag_homogenized": pd.Series(dtype="float64"),
        }
    )

    result = deduplicate_pandas(
        frame,
        tiebreaking_priority=["z_flag_homogenized"],
        group_col="group_id",
    )

    assert result.empty
    assert {"tie_result", "group_id"}.issubset(result.columns)


def test_singleton_and_dangling_neighbor_remain_winners():
    result = _deduplicate(
        [
            _row("singleton", pd.NA),
            _row("dangling", "missing-id"),
        ]
    )

    assert result["tie_result"].astype(int).to_dict() == {
        "singleton": 1,
        "dangling": 1,
    }
    assert result["group_id"].nunique() == 2


def test_catalog_containing_only_stars_keeps_each_star_isolated():
    result = _deduplicate(
        [
            _row("S1", "S2", flag=6),
            _row("S2", "S1", flag=6),
        ]
    )

    assert result["tie_result"].astype(int).to_dict() == {"S1": 3, "S2": 3}
    assert result["group_id"].nunique() == 2


def test_missing_priority_column_fails_clearly():
    with pytest.raises(KeyError, match="unknown_priority"):
        deduplicate_pandas(
            pd.DataFrame([_row("A", "B"), _row("B", "A")]),
            tiebreaking_priority=["unknown_priority"],
        )


def test_missing_semantic_star_flag_fails_clearly():
    frame = pd.DataFrame(
        {
            "CRD_ID": ["A"],
            "compared_to": [pd.NA],
            "z": [0.1],
            "custom_score": [1.0],
        }
    )

    with pytest.raises(KeyError, match="z_flag_homogenized"):
        deduplicate_pandas(
            frame,
            tiebreaking_priority=["custom_score"],
            group_col="group_id",
        )


def test_instrument_priority_mapping_is_required_when_used():
    with pytest.raises(ValueError, match="instrument_type_priority is required"):
        deduplicate_pandas(
            pd.DataFrame([_row("A", "B"), _row("B", "A")]),
            tiebreaking_priority=["instrument_type_homogenized"],
            instrument_type_priority=None,
        )


def test_defined_flag_and_known_instrument_beat_missing_values():
    flag_result = _deduplicate(
        [
            _row("defined", "missing", flag=4),
            _row("missing", "defined", flag=float("nan")),
        ]
    )
    type_result = _deduplicate(
        [
            _row("known", "unknown", instrument="s"),
            _row("unknown", "known", instrument=pd.NA),
        ]
    )

    assert flag_result["tie_result"].astype(int).to_dict() == {
        "defined": 1,
        "missing": 0,
    }
    assert type_result["tie_result"].astype(int).to_dict() == {
        "known": 1,
        "unknown": 0,
    }


def test_defined_redshift_beats_nan_and_all_nan_stays_hard_tie():
    mixed = _deduplicate(
        [
            _row("defined", "missing", z=0.1),
            _row("missing", "defined", z=float("nan")),
        ]
    )
    all_missing = _deduplicate(
        [
            _row("A", "B", z=float("nan")),
            _row("B", "A", z=float("nan")),
        ]
    )

    assert mixed["tie_result"].astype(int).to_dict() == {
        "defined": 1,
        "missing": 0,
    }
    assert all_missing["tie_result"].astype(int).to_dict() == {"A": 2, "B": 2}


def test_canonical_group_id_is_independent_of_row_order():
    rows = [
        _row("A", "B", flag=3),
        _row("B", "A, C", flag=4),
        _row("C", "B", flag=2),
    ]

    forward = _deduplicate(rows).sort_index()
    reverse = _deduplicate(list(reversed(rows))).sort_index()

    assert forward["group_id"].tolist() == reverse["group_id"].tolist()
    assert forward["tie_result"].astype(int).tolist() == reverse[
        "tie_result"
    ].astype(int).tolist()
