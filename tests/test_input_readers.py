import sys
import types
from pathlib import Path

import dask.dataframe as dd
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages"))
if "tables_io" not in sys.modules:
    tables_io = types.ModuleType("tables_io")
    tables_io.types = types.SimpleNamespace(PD_DATAFRAME="PD_DATAFRAME")
    sys.modules["tables_io"] = tables_io

from product_handle import ProductHandle  # noqa: E402


def test_parquet_reader_returns_lazy_dask_dataframe(tmp_path):
    path = tmp_path / "input.parquet"
    pq.write_table(pa.table({"value": [1, 2]}), path)

    result = ProductHandle(path).to_ddf()

    assert isinstance(result, dd.DataFrame)
    assert result.npartitions >= 1
    assert result.compute()["value"].tolist() == [1, 2]


def test_csv_reader_normalizes_numbers_booleans_and_na(tmp_path):
    path = tmp_path / "input.csv"
    path.write_text(
        "number,enabled,optional\n1,true,NA\n2,false,value\n",
        encoding="utf-8",
    )

    result = ProductHandle(path).to_ddf().compute()

    assert result["number"].astype(int).tolist() == [1, 2]
    assert result["enabled"].tolist() == [True, False]
    assert pd.isna(result.loc[0, "optional"])
    assert result.loc[1, "optional"] == "value"


def test_direct_hats_reader_drops_partition_columns(tmp_path):
    payload = tmp_path / "dataset"
    payload.mkdir()
    (tmp_path / "hats.properties").write_text(
        "dataproduct_type=object\n", encoding="utf-8"
    )
    pq.write_table(
        pa.table(
            {
                "CRD_ID": ["A"],
                "value": [10],
                "Norder": [1],
                "Dir": [0],
                "Npix": [3],
            }
        ),
        payload / "part-0.parquet",
    )

    result = ProductHandle(tmp_path).to_ddf()

    assert isinstance(result, dd.DataFrame)
    assert {"Norder", "Dir", "Npix"}.isdisjoint(result.columns)
    assert result.compute()["CRD_ID"].tolist() == ["A"]
