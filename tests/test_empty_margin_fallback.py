import sys
import types
from pathlib import Path

from packaging.version import Version

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages"))
if "tables_io" not in sys.modules:
    tables_io = types.ModuleType("tables_io")
    tables_io.types = types.SimpleNamespace(PD_DATAFRAME="PD_DATAFRAME")
    sys.modules["tables_io"] = tables_io

from product_handle import (  # noqa: E402
    _LEGACY_EMPTY_MARGIN_MESSAGE,
    _is_legacy_empty_margin_error,
    _should_retry_without_margin,
)


def test_recognizes_direct_and_chained_legacy_empty_margin_error():
    direct = ValueError(_LEGACY_EMPTY_MARGIN_MESSAGE)
    assert _is_legacy_empty_margin_error(direct)

    try:
        try:
            raise direct
        except ValueError as error:
            raise RuntimeError("distributed wrapper") from error
    except RuntimeError as wrapped:
        assert _is_legacy_empty_margin_error(wrapped)


def test_retries_exact_error_only_on_legacy_version():
    error = ValueError(_LEGACY_EMPTY_MARGIN_MESSAGE)

    assert _should_retry_without_margin(error, Version("0.7.2"))
    assert not _should_retry_without_margin(error, Version("0.7.3"))
    assert not _should_retry_without_margin(error, Version("0.9.0"))


def test_does_not_retry_unrelated_margin_failure():
    assert not _should_retry_without_margin(
        ValueError("margin schema mismatch"), Version("0.7.2")
    )
