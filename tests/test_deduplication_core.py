import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages"))

from deduplication import (  # noqa: E402
    deduplicate_pandas,
    validate_object_type_inclusion,
)

INSTRUMENT_PRIORITY = {"s": 3, "g": 2, "p": 1}


def _row(
    crd_id,
    compared_to,
    *,
    flag=4.0,
    instrument="s",
    z=0.1,
    object_type=pd.NA,
):
    return {
        "CRD_ID": crd_id,
        "compared_to": compared_to,
        "z": z,
        "z_flag_homogenized": flag,
        "instrument_type_homogenized": instrument,
        "object_type_homogenized": object_type,
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


def test_stars_participate_after_preparation_filters_have_run():
    result = _deduplicate(
        [
            _row("A", "S", flag=4),
            _row("S", "A, B", flag=None, object_type="star"),
            _row("B", "S", flag=3),
        ]
    )

    assert result["tie_result"].astype(int).to_dict() == {"A": 1, "S": 0, "B": 0}
    assert result["group_id"].nunique() == 1


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


def test_catalog_containing_only_stars_does_not_emit_tie_result_three():
    result = _deduplicate(
        [
            _row("S1", "S2", flag=None, object_type="star"),
            _row("S2", "S1", flag=None, object_type="star"),
        ]
    )

    assert result["tie_result"].astype(int).to_dict() == {"S1": 1, "S2": 0}
    assert result["group_id"].nunique() == 1


def test_included_star_without_quality_uses_internal_half_point_score():
    result = deduplicate_pandas(
        pd.DataFrame(
            [
                _row("S", "G", flag=None, object_type="star"),
                _row("G", "S", flag=0, object_type="galaxy"),
            ]
        ),
        tiebreaking_priority=["z_flag_homogenized"],
        delta_z_threshold=0,
    ).set_index("CRD_ID")

    assert result["tie_result"].astype(int).to_dict() == {"S": 1, "G": 0}
    assert pd.isna(result.loc["S", "z_flag_homogenized"])


def test_internal_half_point_loses_to_real_quality_one():
    result = deduplicate_pandas(
        pd.DataFrame(
            [
                _row("S", "G", flag=None, object_type="star"),
                _row("G", "S", flag=1, object_type="galaxy"),
            ]
        ),
        tiebreaking_priority=["z_flag_homogenized"],
        delta_z_threshold=0,
    ).set_index("CRD_ID")

    assert result["tie_result"].astype(int).to_dict() == {"S": 0, "G": 1}


def test_object_type_inclusion_requires_booleans_and_one_enabled_type():
    with pytest.raises(TypeError, match=r"param\.include_star_oth must be a boolean"):
        validate_object_type_inclusion({"include_star_oth": "yes"})
    with pytest.raises(ValueError, match="At least one"):
        validate_object_type_inclusion(
            {
                "include_unclassified_oth": False,
                "include_galaxy_oth": False,
                "include_star_oth": False,
                "include_agn_oth": False,
                "include_qso_oth": False,
                "include_galactic_oth": False,
            }
        )


@pytest.mark.parametrize("object_type", ["star", "galactic"])
def test_default_object_type_policy_is_no_longer_a_dedup_exclusion(object_type):
    result = _deduplicate(
        [
            _row("X", "G", flag=None, object_type=object_type),
            _row("G", "X", flag=4, object_type="galaxy"),
        ]
    )

    assert result.loc["X", "tie_result"] == 0
    assert result.loc["G", "tie_result"] == 1
    assert result["group_id"].nunique() == 1


def test_object_type_inclusion_is_validated_but_not_used_as_dedup_filter():
    result = deduplicate_pandas(
        pd.DataFrame(
            [
                _row("U", "G", flag=4, object_type=pd.NA),
                _row("G", "U", flag=4, object_type="galaxy"),
            ]
        ),
        tiebreaking_priority=["z_flag_homogenized"],
        object_type_inclusion={"include_unclassified_oth": False},
        group_col="group_id",
    ).set_index("CRD_ID")

    assert result["tie_result"].astype(int).to_dict() == {"U": 2, "G": 2}
    assert result["group_id"].nunique() == 1


def _chain_rows():
    return [
        {
            **_row("A", "B", flag=4, z=0.1),
            "ra": 10.0,
            "dec": 0.0,
        },
        {
            **_row("B", "A, C", flag=3, z=0.1),
            "ra": 10.0 + 0.5 / 3600.0,
            "dec": 0.0,
        },
        {
            **_row("C", "B, D", flag=2, z=0.1),
            "ra": 10.0 + 1.0 / 3600.0,
            "dec": 0.0,
        },
        {
            **_row("D", "C", flag=1, z=0.1),
            "ra": 10.0 + 1.5 / 3600.0,
            "dec": 0.0,
        },
    ]


def test_reference_radius_splits_long_transitive_chain():
    result = deduplicate_pandas(
        pd.DataFrame(_chain_rows()),
        tiebreaking_priority=["z_flag_homogenized"],
        max_representative_radius_arcsec=1.0,
        delta_z_threshold=0,
        group_col="group_id",
    ).set_index("CRD_ID")

    assert result.loc[["A", "B", "C"], "group_id"].nunique() == 1
    assert result.loc["D", "group_id"] != result.loc["A", "group_id"]
    assert result["tie_result"].astype(int).to_dict() == {
        "A": 1,
        "B": 0,
        "C": 0,
        "D": 1,
    }


def test_reference_radius_split_is_independent_of_row_order():
    forward = deduplicate_pandas(
        pd.DataFrame(_chain_rows()),
        tiebreaking_priority=["z_flag_homogenized"],
        max_representative_radius_arcsec=1.0,
        delta_z_threshold=0,
        group_col="group_id",
    ).set_index("CRD_ID")
    reverse = deduplicate_pandas(
        pd.DataFrame(list(reversed(_chain_rows()))),
        tiebreaking_priority=["z_flag_homogenized"],
        max_representative_radius_arcsec=1.0,
        delta_z_threshold=0,
        group_col="group_id",
    ).set_index("CRD_ID")

    assert forward["group_id"].to_dict() == reverse["group_id"].to_dict()
    assert forward["tie_result"].astype(int).to_dict() == reverse[
        "tie_result"
    ].astype(int).to_dict()


def test_missing_priority_column_fails_clearly():
    with pytest.raises(KeyError, match="unknown_priority"):
        deduplicate_pandas(
            pd.DataFrame([_row("A", "B"), _row("B", "A")]),
            tiebreaking_priority=["unknown_priority"],
        )


def test_missing_quality_column_fails_clearly():
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
