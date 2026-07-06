import logging
import sys
import types
from pathlib import Path
from unittest.mock import Mock

import dask.dataframe as dd
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages"))
if "tables_io" not in sys.modules:
    tables_io = types.ModuleType("tables_io")
    tables_io.types = types.SimpleNamespace(PD_DATAFRAME="PD_DATAFRAME")
    sys.modules["tables_io"] = tables_io

from specz import (  # noqa: E402
    _normalize_custom_tiebreaking_priorities,
    _requires_z_flag_homogenization,
    validate_combine_configuration,
)
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


def test_semantic_star_flag_is_preserved_without_being_a_priority():
    frame = dd.from_pandas(
        pd.DataFrame(
            {
                "custom_score": [1.0, 2.0],
                "z_flag_homogenized": [4.0, 6.0],
            }
        ),
        npartitions=1,
        sort=False,
    )
    config = {"tiebreaking_priority": ["custom_score"]}

    result, *_ = _homogenize(
        frame,
        config,
        "demo",
        LOGGER,
        type_cast_ok=False,
        require_z_flag_homogenized=True,
    )

    computed = result.compute()
    assert computed["z_flag_homogenized"].tolist() == [4.0, 6.0]
    assert "instrument_type_homogenized" not in computed.columns


@pytest.mark.parametrize(
    ("mode", "cut", "expected"),
    [
        ("concatenate", 0, False),
        ("concatenate", None, False),
        ("concatenate", 3, True),
        ("concatenate", "6", True),
        ("concatenate", 1.5, False),
        ("concatenate", 7, False),
        ("concatenate_and_mark_duplicates", 0, True),
        ("concatenate_and_remove_duplicates", None, True),
    ],
)
def test_flag_homogenization_is_required_by_dedup_or_active_cut(
    mode, cut, expected
):
    assert _requires_z_flag_homogenization(mode, cut) is expected


def test_custom_priority_is_normalized_and_invalid_values_are_logged():
    frame = dd.from_pandas(
        pd.DataFrame({"custom_score": ["1.5", "bad", None, 3]}),
        npartitions=2,
        sort=False,
    )
    logger = Mock()

    result = _normalize_custom_tiebreaking_priorities(
        frame, ["custom_score"], "demo", logger
    ).compute()

    assert str(result["custom_score"].dtype) == "float64"
    assert result["custom_score"].tolist()[0] == 1.5
    assert pd.isna(result["custom_score"].tolist()[1])
    assert pd.isna(result["custom_score"].tolist()[2])
    assert result["custom_score"].tolist()[3] == 3.0
    assert logger.info.call_args.args[6] == 1


def test_custom_priority_must_exist_and_have_numeric_values():
    frame = dd.from_pandas(
        pd.DataFrame({"other": [1, 2], "invalid_score": ["bad", None]}),
        npartitions=1,
        sort=False,
    )

    with pytest.raises(ValueError, match="must exist in every input catalog"):
        _normalize_custom_tiebreaking_priorities(
            frame, ["missing_score"], "demo", LOGGER
        )
    with pytest.raises(ValueError, match="no valid numeric values"):
        _normalize_custom_tiebreaking_priorities(
            frame, ["invalid_score"], "demo", LOGGER
        )


def test_combine_configuration_rejects_invalid_mode_and_empty_dedup_priorities():
    with pytest.raises(ValueError, match="Invalid combine_type"):
        validate_combine_configuration("unknown", ["score"], 0)
    with pytest.raises(ValueError, match="must be non-empty"):
        validate_combine_configuration(
            "concatenate_and_mark_duplicates", [], 0
        )

    mode, priorities = validate_combine_configuration("concatenate", [], 0)
    assert mode == "concatenate"
    assert priorities == []


def test_cut_six_warns_early_for_remove_duplicates():
    logger = Mock()

    validate_combine_configuration(
        "concatenate_and_remove_duplicates",
        ["z_flag_homogenized"],
        6,
        logger,
    )

    logger.warning.assert_called_once()
    assert "final catalog will normally be empty" in logger.warning.call_args.args[0]
