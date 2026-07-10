"""Smoke tests for executable entrypoint imports."""

import importlib.util
import sys
from pathlib import Path


def test_crd_run_module_imports_successfully():
    root = Path(__file__).resolve().parents[1]
    packages = str(root / "packages")
    if packages not in sys.path:
        sys.path.insert(0, packages)

    script = root / "scripts" / "crd-run.py"
    spec = importlib.util.spec_from_file_location("crd_run_import_smoke", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
