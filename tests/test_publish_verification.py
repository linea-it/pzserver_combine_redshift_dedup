import importlib.util
import logging
import sys
import types
from pathlib import Path

import pytest

if "tables_io" not in sys.modules:
    tables_io = types.ModuleType("tables_io")
    tables_io.types = types.SimpleNamespace(PD_DATAFRAME="PD_DATAFRAME")
    sys.modules["tables_io"] = tables_io


def _load_crd_run_module():
    root = Path(__file__).resolve().parents[1]
    packages = str(root / "packages")
    if packages not in sys.path:
        sys.path.insert(0, packages)

    script = root / "scripts" / "crd-run.py"
    spec = importlib.util.spec_from_file_location("crd_run_publish_tests", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_verify_publish_artifacts_detects_corrupted_destination_with_checksum(
    tmp_path,
):
    crd_run = _load_crd_run_module()
    logger = logging.getLogger("test_publish_verification")

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    (src / "catalog.parquet").write_bytes(b"expected bytes")
    (dst / "catalog.parquet").write_bytes(b"corrupted data")

    with pytest.raises(RuntimeError, match="checksum mismatch"):
        crd_run._verify_publish_artifacts(
            [("tree", str(src), str(dst))],
            logger,
            integrity_check=crd_run.PUBLISH_INTEGRITY_CHECK_CHECKSUM,
        )


def test_verify_publish_artifacts_basic_uses_size_only(tmp_path, monkeypatch):
    crd_run = _load_crd_run_module()
    logger = logging.getLogger("test_publish_basic_verification")

    src = tmp_path / "source.txt"
    dst = tmp_path / "published.txt"
    src.write_text("same size", encoding="utf-8")
    dst.write_text("same size", encoding="utf-8")

    monkeypatch.setattr(
        crd_run,
        "_sha256_file",
        lambda path: pytest.fail("basic publish verification should not checksum"),
    )

    crd_run._verify_publish_artifacts(
        [("file", str(src), str(dst))],
        logger,
        integrity_check=crd_run.PUBLISH_INTEGRITY_CHECK_BASIC,
    )


def test_copy_and_verify_publish_artifacts_retries_transient_failure(
    tmp_path, monkeypatch
):
    crd_run = _load_crd_run_module()
    logger = logging.getLogger("test_publish_retry")

    src = tmp_path / "source.txt"
    dst = tmp_path / "published" / "source.txt"
    src.write_text("published content", encoding="utf-8")

    real_verify = crd_run._verify_publish_artifacts
    attempts = {"count": 0}

    def flaky_verify(artifacts, lg, integrity_check):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise FileNotFoundError("simulated stale NFS lookup")
        return real_verify(artifacts, lg, integrity_check)

    monkeypatch.setattr(crd_run, "_verify_publish_artifacts", flaky_verify)
    monkeypatch.setattr(crd_run.time, "sleep", lambda seconds: None)

    crd_run._copy_and_verify_publish_artifacts(
        [("file", str(src), str(dst))],
        logger,
        max_attempts=3,
        retry_delay_seconds=0,
        integrity_check=crd_run.PUBLISH_INTEGRITY_CHECK_BASIC,
    )

    assert attempts["count"] == 2
    assert dst.read_text(encoding="utf-8") == "published content"


def test_runtime_param_config_accepts_publish_integrity_check():
    crd_run = _load_crd_run_module()

    param_config = crd_run._build_runtime_param_config(
        {
            "run": {
                "combine_type": "concatenate",
                "tie_treatment_option": "remove_all",
                "flags_translation_file": "flags_translation.yaml",
            },
            "filters": {
                "z_flag_homogenized_value_to_cut": 0,
                "instrument_type_homogenized": {
                    "include_spectroscopic": True,
                    "include_grism": True,
                    "include_photometric": True,
                    "include_unclassified": True,
                },
                "object_type_homogenized": {
                    "include_unclassified": True,
                    "include_galaxy": True,
                    "include_star": True,
                    "include_agn": True,
                    "include_qso": True,
                    "include_galactic": True,
                },
            },
            "preparation": {
                "repartition_prepared_catalogs": False,
                "prepared_partition_size": "256MB",
            },
            "output": {
                "extra_columns": {},
                "homogenized_columns": {},
                "insert_DP1_footprint_flag": False,
                "insert_rubin_footprint_flag": False,
                "publish_integrity_check": "checksum",
            },
            "diagnostics": {},
        }
    )

    assert (
        param_config["publish_integrity_check"]
        == crd_run.PUBLISH_INTEGRITY_CHECK_CHECKSUM
    )
