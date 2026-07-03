import sys
import types
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages"))
if "tables_io" not in sys.modules:
    tables_io = types.ModuleType("tables_io")
    tables_io.types = types.SimpleNamespace(PD_DATAFRAME="PD_DATAFRAME")
    sys.modules["tables_io"] = tables_io

from product_handle import ProductHandle, _ensure_staged_parquet_non_empty  # noqa: E402


def test_staged_parquet_validation_accepts_rows(tmp_path):
    pq.write_table(pa.table({"value": [1]}), tmp_path / "part-0.parquet")

    _ensure_staged_parquet_non_empty(tmp_path)


@pytest.mark.parametrize("write_empty_file", [False, True])
def test_staged_parquet_validation_rejects_empty_output(
    tmp_path, write_empty_file
):
    if write_empty_file:
        pq.write_table(
            pa.table({"value": pa.array([], type=pa.int64())}),
            tmp_path / "part-0.parquet",
        )

    with pytest.raises(RuntimeError, match="empty after HATS parquet staging"):
        _ensure_staged_parquet_non_empty(tmp_path)


def _write_hats_member(root, name, dataproduct_type, value):
    member = root / name
    payload = member / "dataset"
    payload.mkdir(parents=True)
    (member / "hats.properties").write_text(
        f"dataproduct_type={dataproduct_type}\n", encoding="utf-8"
    )
    pq.write_table(pa.table({"origin": [value]}), payload / "part-0.parquet")


def test_hats_collection_reads_only_object_catalog(tmp_path):
    (tmp_path / "collection.properties").write_text(
        "catalogs=catalog,margin\n", encoding="utf-8"
    )
    _write_hats_member(tmp_path, "catalog", "object", "main")
    _write_hats_member(tmp_path, "margin", "margin", "margin")

    result = ProductHandle(tmp_path).to_ddf().compute()

    assert result["origin"].tolist() == ["main"]


def test_hats_collection_without_object_catalog_fails(tmp_path):
    (tmp_path / "collection.properties").write_text(
        "catalogs=margin\n", encoding="utf-8"
    )
    _write_hats_member(tmp_path, "margin", "margin", "margin")

    with pytest.raises(ValueError, match="dataproduct_type=object"):
        ProductHandle(tmp_path).to_ddf()
