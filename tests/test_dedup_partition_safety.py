import sys
from pathlib import Path
from unittest.mock import Mock, patch

import dask
import dask.dataframe as dd
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages"))

from deduplication import (  # noqa: E402
    SPLIT_REFERENCE_COLUMN,
    _collapse_within_dz,
    _dedup_local_with_margin,
    _log_representative_radius_diagnostics,
    _validate_local_tie_invariants,
    build_global_tie_invariant_diagnostics,
    count_global_edge_group_mismatches,
    count_global_tie_invariant_violations,
)


def _row(crd_id, neighbor, ra, flag, object_type=pd.NA):
    return {
        "CRD_ID": crd_id,
        "compared_to": neighbor,
        "ra": ra,
        "dec": 0.0,
        "z": 0.1,
        "z_flag_homogenized": flag,
        "tie_result": 1,
        "object_type_homogenized": object_type,
    }


def _solve(main, margin):
    return _dedup_local_with_margin(
        pd.DataFrame(main),
        pd.DataFrame(margin),
        pixel=None,
        tiebreaking_priority=["z_flag_homogenized"],
        instrument_type_priority=None,
        group_col="group_id",
        margin_threshold_arcsec=5.0,
    )


def test_boundary_component_has_same_canonical_group_from_both_pixels():
    a = _row("CRD1_1", "CRD1_2", 10.0, 4.0)
    b = _row("CRD1_2", "CRD1_1", 10.0 + 0.3 / 3600.0, 3.0)

    from_a_pixel = _solve([a], [b]).iloc[0]
    from_b_pixel = _solve([b], [a]).iloc[0]

    assert from_a_pixel["group_id"] == from_b_pixel["group_id"]
    assert from_a_pixel["tie_result"] == 1
    assert from_b_pixel["tie_result"] == 0


def test_reference_radius_split_is_stable_across_main_margin_views():
    rows = [
        _row("A", "B", 10.0, 4.0),
        _row("B", "A, C", 10.0 + 0.5 / 3600.0, 3.0),
        _row("C", "B, D", 10.0 + 1.0 / 3600.0, 2.0),
        _row("D", "C", 10.0 + 1.5 / 3600.0, 1.0),
    ]
    first = _dedup_local_with_margin(
        pd.DataFrame(rows[:2]),
        pd.DataFrame(rows[2:]),
        pixel=None,
        tiebreaking_priority=["z_flag_homogenized"],
        instrument_type_priority=None,
        group_col="group_id",
        max_representative_radius_arcsec=1.0,
    ).set_index("CRD_ID")
    second = _dedup_local_with_margin(
        pd.DataFrame(rows[2:]),
        pd.DataFrame(rows[:2]),
        pixel=None,
        tiebreaking_priority=["z_flag_homogenized"],
        instrument_type_priority=None,
        group_col="group_id",
        max_representative_radius_arcsec=1.0,
    ).set_index("CRD_ID")

    assert first.loc["A", "group_id"] == first.loc["B", "group_id"]
    assert second.loc["C", "group_id"] == first.loc["A", "group_id"]
    assert second.loc["D", "group_id"] != first.loc["A", "group_id"]


def test_custom_priority_keeps_excluded_star_outside_graph():
    rows = [
        {
            **_row("A", "B, S", 10.0, 4.0),
            "custom_score": 1.0,
        },
        {
            **_row("B", "A, S", 10.0 + 0.1 / 3600.0, 3.0),
            "custom_score": 10.0,
        },
        {
            **_row("S", "A, B", 10.0 + 0.2 / 3600.0, None, "star"),
            "custom_score": 100.0,
        },
    ]

    result = _dedup_local_with_margin(
        pd.DataFrame(rows),
        pd.DataFrame(),
        pixel=None,
        tiebreaking_priority=["custom_score"],
        instrument_type_priority=None,
        group_col="group_id",
    ).set_index("CRD_ID")

    assert result["tie_result"].astype(int).to_dict() == {"A": 0, "B": 1, "S": 3}
    assert result.loc["A", "group_id"] == result.loc["B", "group_id"]
    assert result.loc["S", "group_id"] != result.loc["B", "group_id"]


def test_local_invariant_accepts_single_winner_and_hard_tie():
    frame = pd.DataFrame(
        {
            "group_id": [1, 1, 2, 2, 3],
            "z_flag_homogenized": [4, 3, 4, 4, pd.NA],
            "tie_result": [1, 0, 2, 2, 3],
        }
    )

    _validate_local_tie_invariants(frame, group_col="group_id", tie_col="tie_result")


def test_global_edge_validation_detects_split_component():
    frame = dd.from_pandas(
        pd.DataFrame(
            {
                "CRD_ID": ["CRD1_1", "CRD1_2"],
                "compared_to": ["CRD1_2", "CRD1_1"],
                "z_flag_homogenized": [4.0, 4.0],
                "group_id": [101, 202],
            }
        ),
        npartitions=2,
        sort=False,
    )

    mismatch, dangling = count_global_edge_group_mismatches(frame)
    mismatch, dangling = dask.compute(mismatch, dangling)

    assert mismatch == 1
    assert dangling == 0


def test_global_edge_validation_detects_dangling_reference():
    frame = dd.from_pandas(
        pd.DataFrame(
            {
                "CRD_ID": ["CRD1_1"],
                "compared_to": ["CRD1_missing"],
                "z_flag_homogenized": [4.0],
                "group_id": [101],
            }
        ),
        npartitions=1,
        sort=False,
    )

    mismatch, dangling = count_global_edge_group_mismatches(frame)
    mismatch, dangling = dask.compute(mismatch, dangling)

    assert mismatch == 0
    assert dangling == 1


def test_delta_z_keeps_separated_extremes_and_drops_bridge():
    mask = pd.Series([True, True, True])
    group = pd.Series([1, 1, 1])
    redshift = pd.Series([0.0010, 0.0015, 0.0020])
    crd_id = pd.Series(["CRD1_1", "CRD1_2", "CRD1_3"])

    survivors = _collapse_within_dz(
        mask, group, redshift, crd_id, threshold=0.001
    )

    assert survivors.tolist() == [True, False, True]


def test_defined_redshift_beats_nan_but_all_nan_group_stays_unresolved():
    mask = pd.Series([True, True, True, True])
    group = pd.Series([1, 1, 2, 2])
    redshift = pd.Series([0.1, float("nan"), float("nan"), float("nan")])
    crd_id = pd.Series(["CRD1_1", "CRD1_2", "CRD2_1", "CRD2_2"])

    survivors = _collapse_within_dz(
        mask, group, redshift, crd_id, threshold=0.001
    )

    assert survivors.tolist() == [True, False, True, True]


def test_global_tie_validation_detects_invalid_patterns():
    frame = dd.from_pandas(
        pd.DataFrame(
            {
                "group_id": [1, 1, 2, 2, 3, 3, 4],
                "z_flag_homogenized": [4, 3, 4, 4, 4, 4, pd.NA],
                "tie_result": [1, 0, 2, 2, 1, 1, 3],
            }
        ),
        npartitions=2,
        sort=False,
    )

    assert count_global_tie_invariant_violations(frame).compute() == 1

    diagnostics, missing_group_rows = build_global_tie_invariant_diagnostics(frame)
    computed = diagnostics.compute()
    assert missing_group_rows.compute() == 0
    assert len(computed) == 1
    assert bool(computed.iloc[0]["multiple_winners"])


def test_global_tie_validation_accepts_compact_labels_without_z_flag():
    labels = dd.from_pandas(
        pd.DataFrame(
            {
                "group_id": [1, 1, 2, 2, 3],
                "tie_result": [1, 0, 2, 2, 3],
            }
        ),
        npartitions=2,
        sort=False,
    )

    assert (
        count_global_tie_invariant_violations(
            labels,
            z_flag_col=None,
        ).compute()
        == 0
    )


def test_representative_radius_diagnostic_warns_for_transitive_chain():
    frame = pd.DataFrame(
        {
            "group_id": [1, 1, 1],
            "tie_result": [1, 0, 0],
            "CRD_ID": ["A", "B", "C"],
            "ra": [10.0, 10.0 + 0.4 / 3600.0, 10.0 + 0.8 / 3600.0],
            "dec": [0.0, 0.0, 0.0],
        }
    )
    logger = Mock()

    with patch("deduplication._phase_logger", return_value=logger):
        diagnostic = _log_representative_radius_diagnostics(
            frame,
            group_col="group_id",
            tie_col="tie_result",
            crd_col="CRD_ID",
            radius_arcsec=0.5,
            partition_tag="test",
        )

    logger.warning.assert_called_once()
    assert "Representative-radius diagnostics" in logger.warning.call_args.args[0]
    assert diagnostic.notna().sum() == 1
    assert diagnostic.max() == pytest.approx(0.8, abs=1e-6)


def test_representative_radius_diagnostic_uses_split_reference_not_winner():
    frame = pd.DataFrame(
        {
            "group_id": [1, 1, 1],
            "tie_result": [0, 1, 0],
            "CRD_ID": ["reference", "winner", "edge"],
            SPLIT_REFERENCE_COLUMN: ["reference"] * 3,
            "ra": [10.0, 10.0 + 0.8 / 3600.0, 10.0 - 0.8 / 3600.0],
            "dec": [0.0, 0.0, 0.0],
        }
    )
    logger = Mock()

    with patch("deduplication._phase_logger", return_value=logger):
        diagnostic = _log_representative_radius_diagnostics(
            frame,
            group_col="group_id",
            tie_col="tie_result",
            crd_col="CRD_ID",
            radius_arcsec=1.0,
            partition_tag="test",
        )

    logger.warning.assert_not_called()
    assert diagnostic.notna().sum() == 1
    assert diagnostic.max() == pytest.approx(0.8, abs=1e-6)
