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


def test_verify_publish_artifacts_detects_corrupted_destination(tmp_path):
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

    def flaky_verify(artifacts, lg):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise FileNotFoundError("simulated stale NFS lookup")
        return real_verify(artifacts, lg)

    monkeypatch.setattr(crd_run, "_verify_publish_artifacts", flaky_verify)
    monkeypatch.setattr(crd_run.time, "sleep", lambda seconds: None)

    crd_run._copy_and_verify_publish_artifacts(
        [("file", str(src), str(dst))],
        logger,
        max_attempts=3,
        retry_delay_seconds=0,
    )

    assert attempts["count"] == 2
    assert dst.read_text(encoding="utf-8") == "published content"
