import sys
from pathlib import Path

import dask.dataframe as dd
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages"))

from product_handle import save_dataframe  # noqa: E402


def _distributed_frame():
    frame = pd.DataFrame(
        {
            "CRD_ID": ["A", "B", "C", "D"],
            "value": pd.Series([1, 2, 3, 4], dtype="Int64"),
            "label": pd.Series(["a", "b", pd.NA, "d"], dtype="string"),
        }
    )
    return frame, dd.from_pandas(frame, npartitions=3, sort=False)


def test_distributed_parquet_export_creates_one_file(tmp_path):
    expected, distributed = _distributed_frame()
    output_base = tmp_path / "catalog"

    save_dataframe(
        distributed,
        output_base,
        "parquet",
        temp_dir=tmp_path / "staging",
    )

    output_path = tmp_path / "catalog.parquet"
    assert output_path.is_file()
    result = pd.read_parquet(output_path).sort_values("CRD_ID").reset_index(drop=True)
    assert result["CRD_ID"].tolist() == expected["CRD_ID"].tolist()
    assert result["value"].astype(int).tolist() == [1, 2, 3, 4]
    assert not list((tmp_path / "staging").rglob("*.parquet"))


def test_distributed_csv_export_creates_one_file_with_one_header(tmp_path):
    expected, distributed = _distributed_frame()
    output_base = tmp_path / "catalog"

    save_dataframe(
        distributed,
        output_base,
        "csv",
        temp_dir=tmp_path / "staging",
    )

    output_path = tmp_path / "catalog.csv"
    assert output_path.is_file()
    assert output_path.read_text(encoding="utf-8").count("CRD_ID,value,label") == 1
    result = pd.read_csv(output_path).sort_values("CRD_ID").reset_index(drop=True)
    assert result["CRD_ID"].tolist() == expected["CRD_ID"].tolist()
    assert result["value"].tolist() == [1, 2, 3, 4]
    assert not list((tmp_path / "staging").rglob("*.parquet"))
