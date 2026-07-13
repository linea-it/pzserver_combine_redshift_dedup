import ast
import logging
import sys
import types
from pathlib import Path
from unittest.mock import Mock

import dask.dataframe as dd
import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages"))
if "tables_io" not in sys.modules:
    tables_io = types.ModuleType("tables_io")
    tables_io.types = types.SimpleNamespace(PD_DATAFRAME="PD_DATAFRAME")
    sys.modules["tables_io"] = tables_io

from specz import (  # noqa: E402
    _apply_instrument_type_filter,
    _apply_object_type_filter,
    _normalize_custom_tiebreaking_priorities,
    _required_homogenized_columns,
    _requires_z_flag_homogenization,
    validate_combine_configuration,
)
from specz_homogenization import (
    _homogenize,  # noqa: E402
    validate_translation_config,  # noqa: E402
)

LOGGER = logging.getLogger("test.homogenization")


def _science_config(**overrides):
    config = {
        "tiebreaking_priority": [],
        "instrument_type_priority": {"s": 3, "g": 2, "p": 1},
        "expr_column_schema": {},
        "translation_rules": {"DEMO": {}},
    }
    for key, value in overrides.items():
        if key == "translation_rules":
            config["translation_rules"] = value
        else:
            config[key] = value
    return config


def test_template_expr_schema_covers_standard_translation_expression_columns():
    root = Path(__file__).resolve().parents[1]
    translation = yaml.safe_load(
        (root / "flags_translation.yaml").read_text(encoding="utf-8")
    )
    protected = {
        "id",
        "ra",
        "dec",
        "z",
        "z_flag",
        "z_err",
        "survey",
        "instrument_type",
        "object_type",
    }
    callables = {"np", "pd", "math", "float", "int", "str", "len"}
    variables = set()
    for ruleset in (translation.get("translation_rules") or {}).values():
        for rule in (ruleset or {}).values():
            if not isinstance(rule, dict):
                continue
            for condition in rule.get("conditions") or []:
                expr = condition.get("expr")
                if not expr:
                    continue
                tree = ast.parse(expr, mode="eval")
                variables.update(
                    node.id
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Name)
                    and node.id not in protected
                    and node.id not in callables
                )

    schema = translation.get("expr_column_schema", {})
    assert variables - set(schema) == set()


def test_object_type_filter_removes_disabled_types(tmp_path):
    ddf = dd.from_pandas(
        pd.DataFrame(
            {
                "CRD_ID": ["G", "S", "U"],
                "object_type_homogenized": ["galaxy", "star", pd.NA],
            }
        ),
        npartitions=1,
    )

    filtered, empty = _apply_object_type_filter(
        ddf,
        param_config={
            "include_unclassified_oth": True,
            "include_galaxy_oth": True,
            "include_star_oth": False,
            "include_agn_oth": True,
            "include_qso_oth": True,
            "include_galactic_oth": False,
        },
        product_name="demo",
        temp_dir=str(tmp_path),
        logger=LOGGER,
    )

    assert not empty
    assert filtered.compute()["CRD_ID"].tolist() == ["G", "U"]


def test_instrument_type_filter_marks_empty_catalog(tmp_path):
    ddf = dd.from_pandas(
        pd.DataFrame(
            {
                "CRD_ID": ["P1", "P2"],
                "instrument_type_homogenized": ["p", pd.NA],
            }
        ),
        npartitions=1,
    )

    filtered, empty = _apply_instrument_type_filter(
        ddf,
        param_config={
            "include_spectroscopic_ith": True,
            "include_grism_ith": True,
            "include_photometric_ith": False,
            "include_unclassified_ith": False,
        },
        product_name="demo",
        temp_dir=str(tmp_path),
        logger=LOGGER,
    )

    assert empty
    assert filtered.compute().empty
    assert (tmp_path / "prepared_demo.empty").read_text().startswith(
        "empty_after_instrument_type_homogenized_filter"
    )


def test_required_homogenized_columns_include_filters_and_priorities():
    required = _required_homogenized_columns(
        param_config={
            "z_flag_homogenized_value_to_cut": 3,
            "include_photometric_ith": False,
            "include_star_oth": False,
        },
        tiebreaking_priority=["custom_rank", "instrument_type_homogenized"],
    )

    assert required == {
        "z_flag_homogenized": ["z_flag_homogenized_value_to_cut"],
        "instrument_type_homogenized": [
            "tiebreaking_priority",
            "instrument_type_filter",
        ],
        "object_type_homogenized": ["object_type_filter"],
    }


def test_translation_schema_rejects_unsafe_expression_with_exact_path():
    config = _science_config(
        translation_rules={
            "DEMO": {
                "object_type_translation": {
                    "conditions": [
                        {"expr": "__import__('os').system('id')", "value": "star"}
                    ]
                }
            }
        }
    )

    with pytest.raises(
        ValueError,
        match=r"translation_rules\.DEMO\.object_type_translation\.conditions\[0\]\.expr",
    ):
        validate_translation_config(config)


def test_translation_schema_rejects_private_attributes_and_typos():
    private = _science_config(
        translation_rules={
            "DEMO": {
                "object_type_translation": {
                    "conditions": [{"expr": "value.__class__", "value": "star"}]
                }
            }
        }
    )
    with pytest.raises(ValueError, match="private attribute"):
        validate_translation_config(private)

    typo = _science_config(
        translation_rules={
            "DEMO": {
                "object_type_translation": {
                    "allow_condition_overlaps": True,
                    "default": None,
                }
            }
        }
    )
    with pytest.raises(ValueError, match="did you mean 'allow_condition_overlap'"):
        validate_translation_config(typo)


def test_translation_schema_validates_output_domains_and_condition_shape():
    invalid_value = _science_config(
        translation_rules={
            "DEMO": {"z_flag_translation": {"default": 9}}
        }
    )
    with pytest.raises(ValueError, match=r"z_flag_translation\.default=9"):
        validate_translation_config(invalid_value)

    invalid_condition = _science_config(
        translation_rules={
            "DEMO": {
                "object_type_translation": {
                    "conditions": [{"expr": "value == 1", "vale": "star"}]
                }
            }
        }
    )
    with pytest.raises(ValueError, match=r"conditions\[0\].*unknown option"):
        validate_translation_config(invalid_condition)


def test_translation_schema_rejects_operational_controls():
    with pytest.raises(ValueError, match="unknown option"):
        validate_translation_config(_science_config(save_expr_columns=False))
    with pytest.raises(ValueError, match="unknown option"):
        validate_translation_config(
            _science_config(tie_invariant_diagnostics_enabled=True)
        )
    with pytest.raises(ValueError, match="unknown option"):
        validate_translation_config(_science_config(repartition_prepared_catalogs=False))


def test_translation_schema_validates_spatial_controls():
    with pytest.raises(ValueError, match="max_representative_radius_arcsec"):
        validate_translation_config(_science_config(max_representative_radius_arcsec=0))

    validate_translation_config(
        _science_config(
            max_representative_radius_arcsec=1.0,
        )
    )


def test_translation_schema_requires_core_scientific_options():
    with pytest.raises(ValueError, match="missing required scientific option"):
        validate_translation_config({})


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
    config = _science_config(
        tiebreaking_priority=["z_flag_homogenized"],
        translation_rules={
            "DEMO": {
                "z_flag_translation": {
                    "default": 0,
                    "1": 2,
                    "conditions": [{"expr": "special", "value": 4}],
                }
            }
        },
    )

    result, *_ = _homogenize(frame, config, "demo", LOGGER, type_cast_ok=False)

    assert result.compute()["z_flag_homogenized"].astype(float).tolist() == [2, 0, 4]


def test_z_flag_fast_path_policy_can_be_disabled_or_required():
    frame = dd.from_pandas(
        pd.DataFrame({"survey": ["demo", "demo"], "z_flag": [0.2, 0.95]}),
        npartitions=1,
        sort=False,
    )
    disabled = _science_config(
        tiebreaking_priority=["z_flag_homogenized"],
        translation_rules={
            "DEMO": {
                "z_flag_translation": {"fast_path": "disabled", "default": 3}
            }
        },
    )
    result, *_ = _homogenize(
        frame, disabled, "demo", LOGGER, type_cast_ok=False
    )
    assert result.compute()["z_flag_homogenized"].tolist() == [3.0, 3.0]

    incompatible = dd.from_pandas(
        pd.DataFrame({"survey": ["demo"], "z_flag": [4]}),
        npartitions=1,
        sort=False,
    )
    required = _science_config(
        tiebreaking_priority=["z_flag_homogenized"],
        translation_rules={
            "DEMO": {
                "z_flag_translation": {"fast_path": "required", "default": 3}
            }
        },
    )
    with pytest.raises(ValueError, match="fast_path is 'required'"):
        _homogenize(incompatible, required, "demo", LOGGER, type_cast_ok=False)


def test_z_flag_fast_path_bypasses_invalid_required_yaml_rule():
    frame = dd.from_pandas(
        pd.DataFrame({"survey": ["demo", "demo"], "z_flag": [0.2, 0.95]}),
        npartitions=1,
        sort=False,
    )
    config = _science_config(
        tiebreaking_priority=["z_flag_homogenized"],
        translation_rules={
            "DEMO": {
                "z_flag_translation": {
                    "conditions": [{"expr": "missing_column > 0", "value": 4}],
                    "default": 0,
                }
            }
        },
    )

    result, *_ = _homogenize(
        frame,
        config,
        "demo",
        LOGGER,
        type_cast_ok=False,
        require_z_flag_homogenized=True,
    )

    assert result.compute()["z_flag_homogenized"].tolist() == [1.0, 3.0]


def test_user_homogenized_column_bypasses_invalid_required_yaml_rule():
    frame = dd.from_pandas(
        pd.DataFrame(
            {"survey": ["demo"], "z_flag_homogenized": [4.0], "z_flag": [99.0]}
        ),
        npartitions=1,
        sort=False,
    )
    config = _science_config(
        tiebreaking_priority=["z_flag_homogenized"],
        translation_rules={
            "DEMO": {
                "z_flag_translation": {
                    "source": "missing_source",
                    1: 4,
                }
            }
        },
    )

    result, *_ = _homogenize(
        frame,
        config,
        "demo",
        LOGGER,
        type_cast_ok=False,
        require_z_flag_homogenized=True,
    )

    assert result.compute()["z_flag_homogenized"].tolist() == [4.0]


def test_required_yaml_translation_fails_on_missing_source_column():
    frame = dd.from_pandas(
        pd.DataFrame({"survey": ["demo"], "z_flag": [99.0]}),
        npartitions=1,
        sort=False,
    )
    config = _science_config(
        translation_rules={
            "DEMO": {
                "z_flag_translation": {
                    "source": "missing_source",
                    1: 4,
                }
            }
        },
    )

    with pytest.raises(ValueError, match="Missing source column 'missing_source'"):
        _homogenize(
            frame,
            config,
            "demo",
            LOGGER,
            type_cast_ok=False,
            require_z_flag_homogenized=True,
        )


def test_required_object_type_yaml_fails_on_invalid_condition():
    frame = dd.from_pandas(
        pd.DataFrame({"survey": ["demo"], "object_type": [pd.NA]}),
        npartitions=1,
        sort=False,
    )
    config = _science_config(
        translation_rules={
            "DEMO": {
                "object_type_translation": {
                    "conditions": [{"expr": "missing_column == 1", "value": "galaxy"}],
                    "default": None,
                }
            }
        },
    )

    with pytest.raises(ValueError, match="Error evaluating condition"):
        _homogenize(
            frame,
            config,
            "demo",
            LOGGER,
            type_cast_ok=False,
            require_object_type_homogenized=True,
        )


def test_instrument_fast_path_can_be_disabled():
    frame = dd.from_pandas(
        pd.DataFrame(
            {"survey": ["demo", "demo"], "instrument_type": [pd.NA] * 2, "type": ["s", "g"]}
        ),
        npartitions=1,
        sort=False,
    )
    config = _science_config(
        tiebreaking_priority=["instrument_type_homogenized"],
        translation_rules={
            "DEMO": {
                "instrument_type_translation": {
                    "fast_path": "disabled",
                    "default": "p",
                }
            }
        },
    )
    result, *_ = _homogenize(frame, config, "demo", LOGGER, type_cast_ok=True)
    assert result.compute()["instrument_type_homogenized"].tolist() == ["p", "p"]


def test_optional_source_absence_keeps_condition_fallback():
    frame = dd.from_pandas(
        pd.DataFrame(
            {"survey": ["demo", "demo"], "object_type": [pd.NA] * 2, "fallback": [1, 0]}
        ),
        npartitions=1,
        sort=False,
    )
    config = _science_config(
        translation_rules={
            "DEMO": {
                "object_type_translation": {
                    "source": "missing_optional_column",
                    "optional_source": True,
                    "STAR": "star",
                    "conditions": [{"expr": "fallback == 1", "value": "galaxy"}],
                    "default": None,
                }
            }
        }
    )
    result, *_ = _homogenize(frame, config, "demo", LOGGER, type_cast_ok=False)
    assert result.compute()["object_type_homogenized"].fillna("missing").tolist() == [
        "galaxy",
        "missing",
    ]


def test_later_condition_wins_and_conflicting_overlap_warns(caplog):
    frame = dd.from_pandas(
        pd.DataFrame(
            {
                "survey": ["demo"],
                "object_type": [pd.NA],
                "score": [2],
            }
        ),
        npartitions=1,
        sort=False,
    )
    config = _science_config(
        translation_rules={
            "DEMO": {
                "object_type_translation": {
                    "conditions": [
                        {"expr": "score > 0", "value": "star"},
                        {"expr": "score > 1", "value": "galaxy"},
                    ],
                    "default": None,
                }
            }
        }
    )

    with caplog.at_level(logging.WARNING, logger=LOGGER.name):
        result, *_ = _homogenize(
            frame, config, "demo", LOGGER, type_cast_ok=False
        )

    assert result.compute()["object_type_homogenized"].tolist() == ["galaxy"]
    assert "later conditions take precedence" in caplog.text


def test_declared_condition_overlap_does_not_warn(caplog):
    frame = dd.from_pandas(
        pd.DataFrame(
            {"survey": ["demo"], "object_type": [pd.NA], "score": [2]}
        ),
        npartitions=1,
        sort=False,
    )
    config = _science_config(
        translation_rules={
            "DEMO": {
                "object_type_translation": {
                    "allow_condition_overlap": True,
                    "conditions": [
                        {"expr": "score > 0", "value": "star"},
                        {"expr": "score > 1", "value": "galaxy"},
                    ],
                    "default": None,
                }
            }
        }
    )

    with caplog.at_level(logging.WARNING, logger=LOGGER.name):
        result, *_ = _homogenize(
            frame, config, "demo", LOGGER, type_cast_ok=False
        )

    assert result.compute()["object_type_homogenized"].tolist() == ["galaxy"]
    assert "later conditions take precedence" not in caplog.text


def test_cosmos_web_flag_translation_compares_normalized_string_type():
    config_path = Path(__file__).resolve().parents[1] / "flags_translation.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["tiebreaking_priority"] = ["z_flag_homogenized"]
    frame = dd.from_pandas(
        pd.DataFrame(
            {
                "survey": ["COSMOS_Web"] * 5,
                "object_type": [pd.NA] * 5,
                "type": ["1", "0", "0", "0", "0"],
                "z_flag": [0, 1, 0, 0, 0],
                "zpdf_med": [1.0, 1.0, 1.01, 1.0, 1.0],
                "zchi2": [1.0, 1.0, 1.0, 1.0, 1.0],
                "nbfilt": [40, 40, 40, 20, 30],
            }
        ),
        npartitions=1,
        sort=False,
    )

    result, *_ = _homogenize(frame, config, "demo", LOGGER, type_cast_ok=False)

    assert result.compute()["z_flag_homogenized"].fillna(-1).tolist() == [
        -1.0,
        0.0,
        1.0,
        2.0,
        3.0,
    ]


def test_object_type_is_always_present_and_may_be_entirely_null():
    frame = dd.from_pandas(
        pd.DataFrame({"survey": ["unknown", "unknown"]}),
        npartitions=1,
        sort=False,
    )

    result, *_ = _homogenize(
        frame, _science_config(), "demo", LOGGER, type_cast_ok=False
    )

    computed = result.compute()
    assert "object_type_homogenized" in computed
    assert computed["object_type_homogenized"].isna().all()


def test_object_type_translation_uses_canonical_renamed_z_flag():
    frame = dd.from_pandas(
        pd.DataFrame(
            {
                "survey": ["demo", "demo", "demo"],
                "object_type": [pd.NA, pd.NA, pd.NA],
                "z_flag": [13, 4, 24],
            }
        ),
        npartitions=1,
        sort=False,
    )
    config = _science_config(
        translation_rules={
            "DEMO": {
                "object_type_translation": {
                    "conditions": [
                        {"expr": "10 <= z_flag < 20", "value": "qso"}
                    ],
                    "default": None,
                }
            }
        }
    )

    result, *_ = _homogenize(frame, config, "demo", LOGGER, type_cast_ok=False)

    assert result.compute()["object_type_homogenized"].tolist() == [
        "qso",
        pd.NA,
        pd.NA,
    ]


def test_object_type_rejects_values_outside_domain():
    valid = dd.from_pandas(
        pd.DataFrame({"object_type_homogenized": ["STAR", "agn", "GALACTIC"]}),
        npartitions=1,
        sort=False,
    )
    result, *_ = _homogenize(
        valid, _science_config(), "demo", LOGGER, type_cast_ok=False
    )
    assert result.compute()["object_type_homogenized"].tolist() == [
        "star",
        "agn",
        "galactic",
    ]

    invalid = dd.from_pandas(
        pd.DataFrame({"object_type_homogenized": ["unknown"]}),
        npartitions=1,
        sort=False,
    )
    with pytest.raises(ValueError, match="Invalid values"):
        _homogenize(invalid, _science_config(), "demo", LOGGER, type_cast_ok=False)


def test_catalog_object_type_rules_use_renamed_flags_and_conservative_defaults():
    config_path = Path(__file__).resolve().parents[1] / "flags_translation.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["tiebreaking_priority"] = []
    frame = dd.from_pandas(
        pd.DataFrame(
            {
                "survey": [
                    "PRIMUS",
                    "VVDS",
                    "VUDS",
                    "VANDELS",
                    "VIPERS_PDR2",
                    "OZDES",
                    "OZDES",
                ],
                "object_type": [pd.NA] * 7,
                "z_flag": [4, 14.5, 23, 214, 213.5, 4, 6],
                "CLASS": ["AGN", pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA],
                "classFlag": [pd.NA, pd.NA, pd.NA, pd.NA, 1, pd.NA, pd.NA],
                "Object_types": [
                    pd.NA,
                    pd.NA,
                    pd.NA,
                    pd.NA,
                    pd.NA,
                    "Photo-z,LRG",
                    "QSO_faint",
                ],
            }
        ),
        npartitions=2,
        sort=False,
    )

    result, *_ = _homogenize(frame, config, "demo", LOGGER, type_cast_ok=False)

    assert result.compute()["object_type_homogenized"].fillna("missing").tolist() == [
        "agn",
        "agn",
        "missing",
        "agn",
        "agn",
        "missing",
        "star",
    ]


def test_3dhst_and_mosdef_object_type_rules_keep_unknowns_null():
    config_path = Path(__file__).resolve().parents[1] / "flags_translation.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["tiebreaking_priority"] = []
    frame = dd.from_pandas(
        pd.DataFrame(
            {
                "survey": ["3D-HST", "3D-HST", "MOSDEF", "MOSDEF", "GAMA_DR4"],
                "object_type": [pd.NA] * 5,
                "z_best_s": [0.0, 1.0, float("nan"), float("nan"), float("nan")],
                "TARGET": [float("nan"), float("nan"), 1.0, 0.0, float("nan")],
            }
        ),
        npartitions=1,
        sort=False,
    )

    result, *_ = _homogenize(frame, config, "demo", LOGGER, type_cast_ok=False)

    assert result.compute()["object_type_homogenized"].fillna("missing").tolist() == [
        "star",
        "missing",
        "missing",
        "missing",
        "missing",
    ]


def test_swire_j2_qso_range_stops_at_15():
    config_path = Path(__file__).resolve().parents[1] / "flags_translation.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["tiebreaking_priority"] = []
    frame = dd.from_pandas(
        pd.DataFrame(
            {
                "survey": ["SWIRE-REVISED"] * 7,
                "object_type": [pd.NA] * 7,
                "mst": [0, 0, -1, 1, -5, 5, -5],
                "J1": [1, 1, 1, 1, 1, 1, 13],
                "J2": [15, 16, 16, 16, 16, 16, 16],
            }
        ),
        npartitions=1,
        sort=False,
    )

    result, *_ = _homogenize(frame, config, "demo", LOGGER, type_cast_ok=False)

    assert result.compute()["object_type_homogenized"].fillna("missing").tolist() == [
        "qso",
        "missing",
        "star",
        "galaxy",
        "missing",
        "missing",
        "qso",
    ]


def test_2df_6df_and_2mrs_use_only_documented_object_classes():
    config_path = Path(__file__).resolve().parents[1] / "flags_translation.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["tiebreaking_priority"] = []
    frame = dd.from_pandas(
        pd.DataFrame(
            {
                "survey": [
                    "2DFLENS",
                    "2DFLENS",
                    "6DFGS",
                    "2MRS",
                    "2MRS",
                    "2MRS",
                ],
                "object_type": [pd.NA] * 6,
                "z_flag": [6, 4, 6, 4, 4, 4],
                "TYPE": [pd.NA, pd.NA, pd.NA, "-5A", "-9", "98"],
            }
        ),
        npartitions=1,
        sort=False,
    )

    result, *_ = _homogenize(frame, config, "demo", LOGGER, type_cast_ok=False)

    assert result.compute()["object_type_homogenized"].fillna("missing").tolist() == [
        "galactic",
        "missing",
        "galactic",
        "galaxy",
        "agn",
        "missing",
    ]


def test_stellar_classification_does_not_overwrite_independent_quality():
    config_path = Path(__file__).resolve().parents[1] / "flags_translation.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["tiebreaking_priority"] = ["z_flag_homogenized"]
    frame = dd.from_pandas(
        pd.DataFrame(
            {
                "survey": [
                    "2DFLENS",
                    "6DFGS",
                    "VIPERS_PDR2",
                    "ELAISS1OID",
                    "ELAISS1OID",
                    "ELAISFBMC",
                    "ELAISFBMC",
                    "SWIRE-REVISED",
                    "SWIRE-REVISED",
                    "SWIRE-REVISED",
                    "OZDES",
                ],
                "z_flag": [6, 6, 4.2, 6, 4, 6, 4, 6, 4, 4, 6],
                "classFlag": [0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0],
                "Class": [0, 0, 0, 2, 4, 0, 0, 0, 0, 0, 0],
                "tSp": [0, 0, 0, 0, 0, 1, 7, 0, 0, 0, 0],
                "mst": [0, 0, 0, 0, 0, 0, 0, 1, -1, -1, 0],
                "J1": [1, 1, 1, 1, 1, 1, 1, 1, 1, 13, 1],
                "J2": [1] * 11,
            }
        ),
        npartitions=2,
        sort=False,
    )

    result, *_ = _homogenize(frame, config, "demo", LOGGER, type_cast_ok=False)

    assert result.compute()["z_flag_homogenized"].fillna(-1).tolist() == [
        -1,
        -1,
        0,
        -1,
        4,
        -1,
        4,
        -1,
        4,
        4,
        -1,
    ]


def test_vimos_uses_only_unambiguous_comm_classifications():
    config_path = Path(__file__).resolve().parents[1] / "flags_translation.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["tiebreaking_priority"] = []
    frame = dd.from_pandas(
        pd.DataFrame(
            {
                "survey": ["VIMOS"] * 5,
                "object_type": [pd.NA] * 5,
                "COMM": [
                    "star",
                    "Star",
                    "Star?",
                    "CIV_[CIII]_(BLAGN)",
                    "Lya(em)_CIV_(BLAGN?)",
                ],
            }
        ),
        npartitions=1,
        sort=False,
    )

    result, *_ = _homogenize(frame, config, "demo", LOGGER, type_cast_ok=False)

    assert result.compute()["object_type_homogenized"].fillna("missing").tolist() == [
        "star",
        "star",
        "missing",
        "agn",
        "missing",
    ]


def test_deimos_remarks_use_only_unambiguous_classifications():
    config_path = Path(__file__).resolve().parents[1] / "flags_translation.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["tiebreaking_priority"] = []
    frame = dd.from_pandas(
        pd.DataFrame(
            {
                "survey": ["DEIMOS_10K"] * 10,
                "object_type": [pd.NA] * 10,
                "z_flag": [4] * 8 + [14, 14],
                "Remarks": [
                    "star",
                    "M star",
                    "Star?",
                    "in halo of bright star",
                    "NaI,TiO,Ha,star",
                    "MgII,QSO?",
                    "CIII],NeIV?(br),QSO",
                    "MgII(br),MgII(abs),[NeV]br,[OII]br(QSO)",
                    "-",
                    "CIV,QSO",
                ],
            }
        ),
        npartitions=1,
        sort=False,
    )

    result, *_ = _homogenize(frame, config, "demo", LOGGER, type_cast_ok=False)

    assert result.compute()["object_type_homogenized"].fillna("missing").tolist() == [
        "star",
        "star",
        "missing",
        "missing",
        "star",
        "missing",
        "qso",
        "qso",
        "agn",
        "qso",
    ]


def test_generic_agn_labels_are_not_promoted_to_qso():
    config_path = Path(__file__).resolve().parents[1] / "flags_translation.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["tiebreaking_priority"] = []
    frame = dd.from_pandas(
        pd.DataFrame(
            {
                "survey": [
                    "HELP-DMU23",
                    "OZDES",
                    "OZDES",
                    "ELAISS1OID",
                    "ELAISS1OID",
                    "ELAISFBMC",
                    "ELAISFBMC",
                ],
                "object_type": [pd.NA] * 7,
                "agn_flag": [1.0] + [float("nan")] * 6,
                "Object_types": [
                    pd.NA,
                    "AGN_reverberation",
                    "AGN_reverberation,XXL_QSO",
                    pd.NA,
                    pd.NA,
                    pd.NA,
                    pd.NA,
                ],
                "z_flag": [4.0] * 7,
                "Class": [float("nan")] * 3 + [1.0, 5.0] + [float("nan")] * 2,
                "tSp": [float("nan")] * 5 + [4.0, 5.0],
            }
        ),
        npartitions=1,
        sort=False,
    )

    result, *_ = _homogenize(frame, config, "demo", LOGGER, type_cast_ok=False)

    assert result.compute()["object_type_homogenized"].tolist() == [
        "agn",
        pd.NA,
        pd.NA,
        "agn",
        "agn",
        "agn",
        "agn",
    ]


def test_vipers_keeps_photometric_star_like_null_but_retains_spectroscopic_blagn():
    config_path = Path(__file__).resolve().parents[1] / "flags_translation.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["tiebreaking_priority"] = []
    frame = dd.from_pandas(
        pd.DataFrame(
            {
                "survey": ["VIPERS_PDR2"] * 3,
                "object_type": [pd.NA] * 3,
                "z_flag": [4.2, 13.2, 213.2],
                "classFlag": [-1, -1, -1],
            }
        ),
        npartitions=1,
        sort=False,
    )

    result, *_ = _homogenize(frame, config, "demo", LOGGER, type_cast_ok=False)

    assert result.compute()["object_type_homogenized"].fillna("missing").tolist() == [
        "missing",
        "agn",
        "agn",
    ]


def test_ozdes_targeting_labels_are_not_treated_as_object_classifications():
    config_path = Path(__file__).resolve().parents[1] / "flags_translation.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["tiebreaking_priority"] = []
    frame = dd.from_pandas(
        pd.DataFrame(
            {
                "survey": ["OZDES"] * 3,
                "object_type": [pd.NA] * 3,
                "Object_types": ["RNDstars", "BrightStar", "WhiteDwarf"],
                "z_flag": [4] * 3,
            }
        ),
        npartitions=1,
        sort=False,
    )

    result, *_ = _homogenize(frame, config, "demo", LOGGER, type_cast_ok=False)

    assert result.compute()["object_type_homogenized"].isna().all()


def test_euclid_optional_source_is_safe_before_column_arrives():
    config_path = Path(__file__).resolve().parents[1] / "flags_translation.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["tiebreaking_priority"] = []
    frame_without_source = dd.from_pandas(
        pd.DataFrame(
            {
                "survey": ["EUCLID_Q1"],
                "object_type": [pd.NA],
            }
        ),
        npartitions=1,
        sort=False,
    )
    missing_result, *_ = _homogenize(
        frame_without_source, config, "demo", LOGGER, type_cast_ok=False
    )
    assert missing_result.compute()["object_type_homogenized"].isna().all()

    frame_with_source = dd.from_pandas(
        pd.DataFrame(
            {
                "survey": ["EUCLID_Q1"] * 5,
                "object_type": [pd.NA] * 5,
                "spe_class": [pd.NA, "STAR", "GALAXY", "QSO", "UNDEF"],
            }
        ),
        npartitions=1,
        sort=False,
    )

    result, *_ = _homogenize(
        frame_with_source, config, "demo", LOGGER, type_cast_ok=False
    )

    assert result.compute()["object_type_homogenized"].fillna("missing").tolist() == [
        "missing",
        "star",
        "galaxy",
        "qso",
        "missing",
    ]


def test_required_homogenized_column_allows_missing_survey_coverage():
    frame = dd.from_pandas(
        pd.DataFrame({"survey": ["known", "missing"], "z_flag": [1, 1]}),
        npartitions=1,
        sort=False,
    )
    config = _science_config(
        tiebreaking_priority=["z_flag_homogenized"],
        translation_rules={
            "KNOWN": {"z_flag_translation": {"default": 1}}
        },
    )

    result, *_ = _homogenize(frame, config, "demo", LOGGER, type_cast_ok=False)

    computed = result.compute()
    assert computed["z_flag_homogenized"].astype(float).fillna(-1).tolist() == [
        1.0,
        -1.0,
    ]


def test_user_homogenized_values_are_normalized_and_validated():
    valid = dd.from_pandas(
        pd.DataFrame({"instrument_type_homogenized": ["S", "g", "P"]}),
        npartitions=1,
        sort=False,
    )
    config = _science_config(
        tiebreaking_priority=["instrument_type_homogenized"]
    )

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
    config = _science_config(tiebreaking_priority=["z_flag_homogenized"])

    with pytest.raises(ValueError, match="Invalid values"):
        _homogenize(frame, config, "demo", LOGGER, type_cast_ok=False)


def test_quality_flag_rejects_legacy_six_and_accepts_all_null():
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
    config = _science_config(tiebreaking_priority=["custom_score"])

    with pytest.raises(ValueError, match="Invalid values"):
        _homogenize(
            frame,
            config,
            "demo",
            LOGGER,
            type_cast_ok=False,
            require_z_flag_homogenized=True,
        )

    all_null = dd.from_pandas(
        pd.DataFrame({"z_flag_homogenized": [pd.NA, pd.NA]}),
        npartitions=1,
        sort=False,
    )
    result, *_ = _homogenize(
        all_null,
        _science_config(tiebreaking_priority=["z_flag_homogenized"]),
        "demo",
        LOGGER,
        type_cast_ok=False,
    )
    assert result.compute()["z_flag_homogenized"].isna().all()


@pytest.mark.parametrize(
    ("mode", "cut", "expected"),
    [
        ("concatenate", 0, False),
        ("concatenate", None, False),
        ("concatenate", 3, True),
        ("concatenate", "6", False),
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


def test_cut_six_has_no_legacy_star_semantics():
    logger = Mock()

    validate_combine_configuration(
        "concatenate_and_remove_duplicates",
        ["z_flag_homogenized"],
        6,
        logger,
    )

    logger.warning.assert_not_called()
