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

from crossmatch_cross import _log_neighbor_saturation  # noqa: E402


class RecordingLogger:
    def __init__(self):
        self.info_calls = []
        self.warning_calls = []

    def info(self, *args):
        self.info_calls.append(args)

    def warning(self, *args):
        self.warning_calls.append(args)


def _saturated_pairs():
    return pd.DataFrame(
        {
            "sourceleft": ["survey", "survey", "survey", "survey"],
            "CRD_IDleft": ["A", "A", "B", "B"],
            "CRD_IDright": ["X", "Y", "X", "Y"],
        }
    )


def test_saturation_warns_using_all_source_objects_as_denominator():
    logger = RecordingLogger()

    _log_neighbor_saturation(
        _saturated_pairs(),
        id_col="CRD_IDleft",
        source_col="sourceleft",
        limit=2,
        logger=logger,
        context="test",
        total_by_source={"survey": 100},
        warn_fraction=0.01,
        fail_fraction=0.10,
    )

    assert len(logger.warning_calls) == 1
    assert len(logger.info_calls) == 1
    assert "returned-match diagnostics" in logger.info_calls[0][0]


def test_saturation_can_fail_or_be_disabled():
    kwargs = dict(
        pairs=_saturated_pairs(),
        id_col="CRD_IDleft",
        source_col="sourceleft",
        limit=2,
        context="test",
        total_by_source={"survey": 100},
        warn_fraction=0.01,
    )

    with pytest.raises(RuntimeError, match="increase crossmatch_n_neighbors"):
        _log_neighbor_saturation(
            logger=RecordingLogger(), fail_fraction=0.02, **kwargs
        )

    _log_neighbor_saturation(
        logger=RecordingLogger(), fail_fraction=None, **kwargs
    )
