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

from specz_homogenization import _homogenize  # noqa: E402


LOGGER = logging.getLogger("test.homogenization")


def test_yaml_flag_translation_applies_direct_default_and_condition():
    frame = dd.from_pandas(
        pd.DataFrame(
            {
                "survey": ["demo", "demo", "demo"],
                "z_flag": [1, 9, 2],
                "special": [False, False, True],
            }
        ),
        npartitions=2,
        sort=False,
    )
    config = {
        "tiebreaking_priority": ["z_flag_homogenized"],
        "translation_rules": {
            "DEMO": {
                "z_flag_translation": {
                    "default": 0,
                    "1": 2,
                    "conditions": [{"expr": "special", "value": 4}],
                }
            }
        },
    }

    result, *_ = _homogenize(frame, config, "demo", LOGGER, type_cast_ok=False)

    assert result.compute()["z_flag_homogenized"].astype(float).tolist() == [2, 0, 4]


def test_homogenization_requires_translation_for_every_survey():
    frame = dd.from_pandas(
        pd.DataFrame({"survey": ["known", "missing"], "z_flag": [1, 1]}),
        npartitions=1,
        sort=False,
    )
    config = {
        "tiebreaking_priority": ["z_flag_homogenized"],
        "translation_rules": {
            "KNOWN": {"z_flag_translation": {"default": 1}}
        },
    }

    with pytest.raises(ValueError, match="MISSING"):
        _homogenize(frame, config, "demo", LOGGER, type_cast_ok=False)


def test_user_homogenized_values_are_normalized_and_validated():
    valid = dd.from_pandas(
        pd.DataFrame({"instrument_type_homogenized": ["S", "g", "P"]}),
        npartitions=1,
        sort=False,
    )
    config = {"tiebreaking_priority": ["instrument_type_homogenized"]}

    result, *_ = _homogenize(valid, config, "demo", LOGGER, type_cast_ok=False)
    assert result.compute()["instrument_type_homogenized"].tolist() == ["s", "g", "p"]

    invalid = dd.from_pandas(
        pd.DataFrame({"instrument_type_homogenized": ["unknown"]}),
        npartitions=1,
        sort=False,
    )
    with pytest.raises(ValueError, match="Invalid values"):
        _homogenize(invalid, config, "demo", LOGGER, type_cast_ok=False)


def test_user_homogenized_flag_rejects_values_outside_domain():
    frame = dd.from_pandas(
        pd.DataFrame({"z_flag_homogenized": [4, 5]}),
        npartitions=1,
        sort=False,
    )
    config = {"tiebreaking_priority": ["z_flag_homogenized"]}

    with pytest.raises(ValueError, match="Invalid values"):
        _homogenize(frame, config, "demo", LOGGER, type_cast_ok=False)
