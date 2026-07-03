import sys
from pathlib import Path

import pandas as pd
import dask.dataframe as dd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages"))

from crossmatch_diagnostics import (  # noqa: E402
    compute_projected_pairs,
    log_component_size_diagnostics,
    log_neighbor_count_diagnostics,
    log_pair_separation_diagnostics,
    project_catalog_for_pair_crossmatch,
)


class RecordingLogger:
    def __init__(self):
        self.info_calls = []

    def info(self, *args):
        self.info_calls.append(args)


class FakeCatalog:
    def __init__(self, frame, margin=None):
        self.frame = frame
        self.columns = frame.columns
        self.margin = margin

    def __getitem__(self, columns):
        return FakeCatalog(self.frame.loc[:, columns].copy())


def test_pair_catalog_projection_is_narrow_and_preserves_margin():
    frame = pd.DataFrame(
        {
            "CRD_ID": ["A"],
            "ra": [1.0],
            "dec": [2.0],
            "source": ["survey"],
            "unused": ["large"],
        }
    )
    catalog = FakeCatalog(frame, margin=FakeCatalog(frame.copy()))

    projected = project_catalog_for_pair_crossmatch(catalog, include_source=True)

    assert projected.columns.tolist() == ["CRD_ID", "ra", "dec", "source"]
    assert projected.margin.columns.tolist() == ["CRD_ID", "ra", "dec", "source"]


def test_compute_projected_pairs_transfers_only_requested_plain_columns():
    source = dd.from_pandas(
        pd.DataFrame(
            {
                "CRD_IDleft": ["A", "B"],
                "CRD_IDright": ["C", "D"],
                "sourceleft": ["one", "two"],
                "large_unused_column": ["x" * 100, "y" * 100],
            }
        ),
        npartitions=2,
    )

    result = compute_projected_pairs(
        source,
        ["CRD_IDleft", "CRD_IDright", "sourceleft"],
    )

    assert type(result) is pd.DataFrame
    assert result.columns.tolist() == ["CRD_IDleft", "CRD_IDright", "sourceleft"]
    assert result["CRD_IDleft"].tolist() == ["A", "B"]


def test_pair_diagnostics_exclude_self_matches_and_log_radius_fractions():
    logger = RecordingLogger()
    pairs = pd.DataFrame(
        {
            "left": ["A", "A", "B", "C"],
            "right": ["A", "B", "C", "D"],
            "_dist_arcsec": [0.0, 0.1, 0.3, 0.49],
        }
    )

    log_pair_separation_diagnostics(
        pairs,
        left_col="left",
        right_col="right",
        radius_arcsec=0.5,
        logger=logger,
        context="test",
    )

    message, *args = logger.info_calls[0]
    assert "separation diagnostics" in message
    assert args[1] == 3


def test_neighbor_and_component_diagnostics_log_distributions():
    logger = RecordingLogger()

    log_neighbor_count_diagnostics(
        pd.Series([1, 2, 5, 10]),
        limit=10,
        logger=logger,
        context="test",
    )
    log_component_size_diagnostics(
        {
            "A": {"B"},
            "B": {"A", "C"},
            "C": {"B"},
            "D": {"E"},
            "E": {"D"},
        },
        logger=logger,
        context="test",
    )

    assert "returned-match diagnostics" in logger.info_calls[0][0]
    assert "component-size diagnostics" in logger.info_calls[1][0]
    assert "low_density" in logger.info_calls[1][0]
    assert logger.info_calls[1][1:4] == ("test", 2, 5)
