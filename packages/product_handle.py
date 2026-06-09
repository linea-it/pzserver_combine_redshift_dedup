"""File ingestion helpers for CRC products.

Provides a unified interface to read Parquet/CSV/HDF5/FITS/plain-text files
and HATS catalogs, returning Dask DataFrames with normalized dtypes.

Public API:
    - ProductHandle
"""

# -----------------------
# Standard library
# -----------------------
import csv
import glob
import io
import json
import os
import shutil
from pathlib import Path

# -----------------------
# Third-party
# -----------------------
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import tables_io
from astropy.table import Table

__all__ = ["ProductHandle", "build_collection_with_retry", "save_dataframe"]


# -----------------------
# Constants
# -----------------------
# NA tokens normalized to pd.NA after decoding from bytes/text
_NA_TOKENS = {"", "na", "nan", "null", "none", "<na>"}
_HATS_DTYPE_STR = pd.ArrowDtype(pa.string())
_HATS_DTYPE_FLOAT = pd.ArrowDtype(pa.float64())
_HATS_DTYPE_INT = pd.ArrowDtype(pa.int64())
_HATS_DTYPE_INT8 = pd.ArrowDtype(pa.int8())


# -----------------------
# Reader: unified handle + format-specific readers
# -----------------------
class ProductHandle:
    """Unified interface to read Parquet/CSV/HDF5/FITS/plain text/HATS into a Dask DataFrame."""

    def __init__(self, filepath):
        """Initialize.

        Args:
            filepath: Path to a data file.
        """
        self.filepath = Path(filepath)

        # Handle compound extensions like ".csv.gz"
        full_suffix = "".join(self.filepath.suffixes).lower()
        if full_suffix.endswith((".csv.gz", ".csv.bz2", ".csv.xz", ".csv.zst")):
            self.base_ext = ".csv"
        elif full_suffix.endswith((".tsv.gz", ".tsv.bz2", ".tsv.xz", ".tsv.zst")):
            self.base_ext = ".tsv"
        elif full_suffix.endswith((".txt.gz", ".txt.bz2", ".txt.xz", ".txt.zst")):
            self.base_ext = ".txt"
        else:
            self.base_ext = self.filepath.suffix.lower()

        self.delimiter = None
        self.has_header = True
        self.column_names = None

    # -----------------------
    # Public API
    # -----------------------
    def to_ddf(self):
        """Read the file and return a Dask DataFrame."""
        if self.filepath.is_dir():
            hats_root = self._resolve_hats_catalog_root(self.filepath)
            if hats_root:
                return self._read_hats_to_ddf(hats_root)
            raise ValueError(f"Unsupported directory input: {self.filepath}")

        if self.base_ext == ".parquet":
            return dd.read_parquet(self.filepath)

        elif self.base_ext in {".csv", ".tsv", ".txt"}:
            self._inspect_csv()

            ddf_raw = dd.read_csv(
                self.filepath,
                delimiter=self.delimiter,
                header=0 if self.has_header else None,
                names=self.column_names if not self.has_header else None,
                dtype="object",
                na_values=["", "NA", "NaN", "null", "None"],
                keep_default_na=True,
                comment="#",
                encoding_errors="replace",
                compression="infer",
            )

            sample_rows = 200_000
            pdf_sample = pd.read_csv(
                self.filepath,
                delimiter=self.delimiter,
                header=0 if self.has_header else None,
                names=self.column_names if not self.has_header else None,
                dtype="object",
                na_values=["", "NA", "NaN", "null", "None"],
                keep_default_na=True,
                comment="#",
                low_memory=False,
                nrows=sample_rows,
            )

            schema = _infer_csv_schema(pdf_sample, frac_threshold=0.995)
            meta = _meta_from_schema(ddf_raw.columns, schema)

            ddf = ddf_raw.map_partitions(_apply_csv_schema, schema, meta=meta)
            return ddf

        elif self.base_ext in {".fits", ".fit"}:
            return self._read_fits_to_ddf()

        elif self.base_ext in {".h5", ".hdf5"}:
            return self._read_hdf5_to_ddf()

        else:
            raise ValueError(f"Unsupported file extension: {self.filepath}")

    def _resolve_hats_catalog_root(self, path: Path) -> Path | None:
        """Resolve HATS root using collection/object properties priority."""
        max_depth = 3
        candidates = [
            p
            for p in path.rglob("*")
            if p.is_file() and self._relative_depth(path, p) <= max_depth
        ]

        collection_candidates = [
            p for p in candidates if p.name.lower() == "collection.properties"
        ]
        if collection_candidates:
            return sorted(collection_candidates)[0].parent

        hats_properties = [
            p for p in candidates if p.name.lower() == "hats.properties"
        ]
        for candidate in sorted(hats_properties):
            if self._is_object_hats_properties_file(candidate):
                return candidate.parent

        legacy_properties = [
            p for p in candidates if p.name.lower() == "properties"
        ]
        for candidate in sorted(legacy_properties):
            if self._is_object_hats_properties_file(candidate):
                return candidate.parent

        return None

    def _relative_depth(self, root_path: Path, file_path: Path) -> int:
        return len(file_path.relative_to(root_path).parts) - 1

    def _is_object_hats_properties_file(self, filepath: Path) -> bool:
        try:
            with filepath.open("rb") as handle:
                text = handle.read(262144).decode("utf-8", errors="ignore")
        except Exception:
            return False
        return self._properties_value(text, "dataproduct_type") == "object"

    def _properties_value(self, text: str, key_name: str) -> str | None:
        wanted = key_name.lower()
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key = line.split("=", 1)[0].strip().lower()
            if not key:
                continue
            if key == wanted:
                return line.split("=", 1)[1].strip().lower()
        return None

    def _read_hats_to_ddf(self, hats_root: Path):
        """Read the object catalog parquet payload from a HATS directory."""
        object_root = self._resolve_object_catalog_root(hats_root) or hats_root
        read_path = self._resolve_hats_parquet_payload_path(object_root)

        ddf = dd.read_parquet(read_path)

        hats_partition_cols = {"Norder", "Dir", "Npix"}
        drop_cols = [col for col in hats_partition_cols if col in ddf.columns]
        if drop_cols:
            ddf = ddf.drop(columns=drop_cols, errors="ignore")

        return ddf

    def _resolve_hats_parquet_payload_path(self, object_root: Path) -> Path:
        """Resolve the parquet payload directory inside a HATS object catalog."""
        candidates = [p for p in object_root.iterdir() if p.is_dir()]

        metadata_candidates = [
            p for p in candidates
            if (p / "_metadata").exists() or (p / "_common_metadata").exists()
        ]
        if metadata_candidates:
            return sorted(metadata_candidates)[0]

        partition_candidates = [
            p for p in candidates
            if any(child.is_dir() and child.name.startswith("Norder=") for child in p.iterdir())
        ]
        if partition_candidates:
            return sorted(partition_candidates)[0]

        if any(object_root.glob("*.parquet")):
            return object_root

        recursive_parquet = sorted(object_root.rglob("*.parquet"))
        if recursive_parquet:
            return recursive_parquet[0].parent

        raise ValueError(f"No parquet payload found inside HATS object catalog: {object_root}")

    def _infer_hats_data_columns(self, catalog_path: Path):
        """Infer HATS data columns from parquet schema, excluding partition columns."""
        hats_partition_cols = {"Norder", "Dir", "Npix"}
        object_root = self._resolve_object_catalog_root(catalog_path) or catalog_path
        parquet_files = sorted(object_root.rglob("*.parquet"))
        if not parquet_files:
            return None
        schema = pq.read_schema(parquet_files[0])
        columns = [name for name in schema.names if name not in hats_partition_cols]
        return columns or None

    def _resolve_object_catalog_root(self, catalog_path: Path) -> Path | None:
        candidates = [
            p
            for p in catalog_path.rglob("*")
            if p.is_file()
            and p.name.lower() in {"hats.properties", "properties"}
            and self._relative_depth(catalog_path, p) <= 3
        ]
        for candidate in sorted(candidates, key=lambda p: (p.name.lower() != "hats.properties", str(p))):
            if self._is_object_hats_properties_file(candidate):
                return candidate.parent
        return None

    # -----------------------
    # CSV helpers
    # -----------------------
    def _inspect_csv(self):
        """Detect delimiter/header using robust heuristics."""
        with open(self.filepath, "rb") as fb:
            raw = fb.read(128 * 1024)
        sample = raw.decode("utf-8", errors="replace")

        # Delimiter
        try:
            sniffed = csv.Sniffer().sniff(sample, delimiters=[",", "\t", ";", "|"])
            self.delimiter = sniffed.delimiter
        except Exception:
            if "\t" in sample and "," not in sample:
                self.delimiter = "\t"
            elif ";" in sample and "," not in sample:
                self.delimiter = ";"
            else:
                self.delimiter = ","

        # Header?
        try:
            df_head = pd.read_csv(
                io.StringIO(sample), delimiter=self.delimiter, nrows=5, header=None
            )
        except Exception:
            df_head = pd.DataFrame()

        def _looks_like_header(df0: pd.DataFrame) -> bool:
            if df0.empty:
                return False
            row0 = df0.iloc[0].astype(str).tolist()

            def is_number(s: str) -> bool:
                try:
                    float(s)
                    return True
                except Exception:
                    return False

            non_numeric = sum(1 for v in row0 if not is_number(v))
            return non_numeric >= max(1, len(row0) // 2)

        if not df_head.empty and _looks_like_header(df_head):
            self.has_header = True
        else:
            self.has_header = False
            self.column_names = [
                f"col_{i}" for i in range(df_head.shape[1] if not df_head.empty else 0)
            ]

    # -----------------------
    # FITS reader (+ postprocess)
    # -----------------------
    def _read_fits_to_ddf(self):
        """Read FITS to Dask DataFrame with nullable semantics preserved."""
        table = Table.read(self.filepath)

        # 1) Pandas DF with NAs preserved (including boolean with pd.NA where masked)
        df = _astropy_table_to_pandas_nullable(table)

        # 2) Decode bytes/strings, normalize NA tokens, and promote text-boolean to BooleanDtype
        df = _normalize_after_read(df, source="fits")

        # 3) Dynamic check: does this Dask preserve pandas BooleanDtype with NA?
        def _dask_preserves_nullable_boolean() -> bool:
            try:
                probe = pd.DataFrame(
                    {"_p": pd.Series([True, pd.NA, False], dtype="boolean")}
                )
                res = dd.from_pandas(probe, npartitions=1).compute()
                return (str(res["_p"].dtype) == "boolean") and bool(
                    res["_p"].isna().iloc[1]
                )
            except Exception:
                return False

        preserves = _dask_preserves_nullable_boolean()

        if not preserves:
            # Stable fallback: re-encode booleans as Int8 (1/0/<NA>)
            for c in df.columns:
                s = df[c]
                if pd.api.types.is_bool_dtype(s) and str(s.dtype) == "boolean":
                    df[c] = s.map({True: 1, False: 0}).astype("Int8")

        # 4) Lift to Dask
        ddf = dd.from_pandas(df, npartitions=1)

        return ddf

    # -----------------------
    # HDF5 reader (+ postprocess)
    # -----------------------
    def _read_hdf5_to_ddf(self):
        """Read HDF5 to Dask DataFrame via tables_io with dtype normalization."""
        df = tables_io.read(self.filepath, tables_io.types.PD_DATAFRAME)
        df = _normalize_after_read(df, source="hdf5")
        return dd.from_pandas(df, npartitions=1)


# -----------------------
# Reader helpers (CSV + decode/normalize + astropy conversions)
# -----------------------
def _string_bool_to_boolean(series: pd.Series) -> pd.Series:
    """Convert 'true'/'false'/NA to pandas BooleanDtype without evaluating NA."""
    s = series.astype("string")

    def _parse(x):
        if pd.isna(x):
            return pd.NA
        t = str(x).strip().lower()
        if t == "true":
            return True
        if t == "false":
            return False
        return pd.NA

    return s.map(_parse).astype("boolean")


def _normalize_na_tokens_frame(pdf: pd.DataFrame) -> pd.DataFrame:
    """Normalize common NA tokens to pd.NA on string/object columns."""
    pdf = pdf.copy()
    for c in pdf.columns:
        s = pdf[c]
        if s.dtype == "object" or pd.api.types.is_string_dtype(s):
            pdf[c] = s.map(
                lambda x: (
                    pd.NA
                    if (
                        x is None
                        or (isinstance(x, str) and x.strip().lower() in _NA_TOKENS)
                    )
                    else x
                )
            )
    return pdf


def _csv_postprocess(pdf: pd.DataFrame) -> pd.DataFrame:
    """Normalize NA tokens and stabilize dtypes for CSV-loaded frames."""
    pdf = _normalize_na_tokens_frame(pdf)
    pdf = pdf.convert_dtypes(dtype_backend="numpy_nullable", convert_boolean=False)
    return pdf


def _infer_csv_schema(pdf: pd.DataFrame, *, frac_threshold: float = 0.995) -> dict:
    """Infer per-column target dtypes for CSV (boolean/Int64/Float64/string)."""
    schema = {}
    for c in pdf.columns:
        s = pdf[c]

        # Only attempt inference on text/object-like columns
        if not (s.dtype == "object" or pd.api.types.is_string_dtype(s)):
            schema[c] = "string"
            continue

        # Normalize NA tokens and ensure string dtype for checks
        s2 = s.map(
            lambda x: (
                pd.NA
                if (
                    x is None
                    or (isinstance(x, str) and x.strip().lower() in _NA_TOKENS)
                )
                else x
            )
        )
        s2 = s2.astype("string").map(lambda x: x.strip() if isinstance(x, str) else x)

        # Strict textual boolean detection
        s2_lower = s2.str.lower()
        non_na = s2_lower.dropna()
        if non_na.empty:
            schema[c] = "string"
            continue
        if set(non_na.unique()) <= {"true", "false"}:
            schema[c] = "boolean"
            continue

        # Numeric inference
        num = pd.to_numeric(s2, errors="coerce")
        nn_total = int(num.notna().sum())
        if nn_total == 0:
            schema[c] = "string"
            continue

        frac_numeric = nn_total / max(1, int(s2.notna().sum()))
        if frac_numeric >= frac_threshold:
            non_na_num = num.dropna()
            has_frac = (non_na_num % 1 != 0).any()
            schema[c] = "Float64" if has_frac else "Int64"
        else:
            schema[c] = "string"
    return schema


def _apply_csv_schema(pdf: pd.DataFrame, schema: dict) -> pd.DataFrame:
    """Apply inferred CSV schema column-wise without coercing 0/1 to booleans."""
    out = {}
    for c in pdf.columns:
        s = pdf[c]
        # Only transform string/object-like columns; keep others as-is
        if s.dtype != "object" and not pd.api.types.is_string_dtype(s):
            out[c] = s
            continue

        # Normalize NA tokens and trim
        s2 = s.map(
            lambda x: (
                pd.NA
                if (
                    x is None
                    or (isinstance(x, str) and x.strip().lower() in _NA_TOKENS)
                )
                else (x.strip() if isinstance(x, str) else x)
            )
        )
        target = schema.get(c, "string")

        if target == "Int64":
            out[c] = pd.to_numeric(s2, errors="coerce").astype("Int64")
        elif target == "Float64":
            out[c] = pd.to_numeric(s2, errors="coerce").astype("Float64")
        elif target == "boolean":
            out[c] = _string_bool_to_boolean(s2)
        else:
            out[c] = s2.astype("string")
    return pd.DataFrame(out, columns=pdf.columns)


def _meta_from_schema(columns, schema: dict) -> pd.DataFrame:
    """Build an empty DataFrame matching the given schema for Dask meta."""
    data = {}
    for c in columns:
        t = schema.get(c, "string")
        if t == "Int64":
            data[c] = pd.Series([], dtype="Int64")
        elif t == "Float64":
            data[c] = pd.Series([], dtype="Float64")
        elif t == "boolean":
            data[c] = pd.Series([], dtype="boolean")
        else:
            data[c] = pd.Series([], dtype="string")
    return pd.DataFrame(data)


def _is_fixed_bytes(s: pd.Series) -> bool:
    """Return True if dtype is fixed-width bytes (NumPy kind 'S')."""
    return getattr(s.dtype, "kind", "") == "S"


def _is_object_bytes_series(s: pd.Series, sample_size: int = 256) -> bool:
    """Return True if dtype is object and most non-null values are bytes-like."""
    if s.dtype != "object":
        return False
    non_null = s.dropna()
    if non_null.empty:
        return True
    sample = non_null.head(sample_size)
    n = len(sample)
    cnt = sum(isinstance(v, (bytes, bytearray, np.bytes_)) for v in sample)
    return cnt / max(1, n) >= 0.8


def _normalize_na_tokens_to_pdna(s: pd.Series) -> pd.Series:
    """Normalize common NA tokens (case-insensitive) to pd.NA on a string Series."""
    if not pd.api.types.is_string_dtype(s):
        s = s.astype("string")
    return s.map(
        lambda x: pd.NA if x is None or str(x).strip().lower() in _NA_TOKENS else x
    )


def _decode_byteslike_to_string(
    series: pd.Series, *, encoding: str, errors: str = "replace"
) -> pd.Series:
    """Decode fixed-width or object-bytes to pandas StringDtype and normalize NA tokens."""
    if _is_fixed_bytes(series):
        it = series.astype("O")
        as_str = it.map(
            lambda b: (
                b.decode(encoding, errors=errors)
                if isinstance(b, (bytes, bytearray))
                else ("" if b is None else str(b))
            )
        )
    elif _is_object_bytes_series(series):
        as_str = series.map(
            lambda b: (
                b.decode(encoding, errors=errors)
                if isinstance(b, (bytes, bytearray, np.bytes_))
                else ("" if b is None else str(b))
            )
        )
    else:
        return series

    as_str = as_str.map(lambda x: x.strip() if isinstance(x, str) else x).astype(
        "string"
    )
    as_str = _normalize_na_tokens_to_pdna(as_str)
    return as_str


def _looks_like_bool_text(series: pd.Series) -> bool:
    """Return True if non-NA values are a subset of {'true','false'}."""
    if series.empty:
        return False
    sample = series.dropna()
    if sample.empty:
        return False
    vals = set(sample.astype(str).str.strip().str.lower().tolist())
    return vals.issubset({"true", "false"})


def _bytes_bool_to_boolean(series: pd.Series) -> pd.Series:
    """Convert textual booleans 'true'/'false'/NA to pandas BooleanDtype safely."""
    s = series.astype("string")
    s_norm = s.str.strip().str.lower()

    out = pd.Series(pd.NA, index=s.index, dtype="boolean")
    out = out.mask(s_norm == "true", True)
    out = out.mask(s_norm == "false", False)
    return out


def _normalize_after_read(df: pd.DataFrame, *, source: str) -> pd.DataFrame:
    """Normalize decoded bytes/strings, NA tokens, and strict text booleans after read."""
    df = df.copy()
    text_encoding = "ascii" if source == "fits" else "utf-8"

    for c in df.columns:
        s = df[c]

        # Case 1: bytes-like -> decode
        was_bytes = _is_fixed_bytes(s) or _is_object_bytes_series(s)
        if was_bytes:
            decoded = _decode_byteslike_to_string(
                s, encoding=text_encoding, errors="replace"
            )
            if _looks_like_bool_text(decoded):
                df[c] = _bytes_bool_to_boolean(decoded)
            else:
                df[c] = decoded
            continue

        # Case 2: string-like -> normalize NA tokens and strict boolean text
        if pd.api.types.is_string_dtype(s):
            s_norm = _normalize_na_tokens_to_pdna(s.astype("string"))
            if _looks_like_bool_text(s_norm):
                df[c] = _string_bool_to_boolean(s_norm)
            else:
                df[c] = s_norm
            continue

        # Other dtypes: keep

    try:
        df = df.convert_dtypes(dtype_backend="numpy_nullable", convert_boolean=False)
    except Exception:
        pass
    return df


def _astropy_table_to_pandas_nullable(table: Table) -> pd.DataFrame:
    """Convert Astropy Table to pandas with nullable dtypes and masked values as NA."""
    out = {}
    for name in table.colnames:
        col = table[name]
        has_mask = hasattr(col, "mask")

        # FITS LOGICAL -> pandas nullable boolean with NA preserved
        if getattr(col.dtype, "kind", None) == "b":
            s = pd.Series(col.astype(object))
            if has_mask:
                s.loc[np.asarray(col.mask, dtype=bool)] = pd.NA
            out[name] = s.astype("boolean")
            continue

        if has_mask:
            s = pd.Series(col.astype(object))
            s.loc[np.asarray(col.mask, dtype=bool)] = pd.NA

            # Try numeric coercion
            s_try = pd.to_numeric(s, errors="coerce")
            nn = s.notna().sum()
            if nn > 0:
                frac_num = s_try.notna().sum() / nn
                if frac_num >= 0.995:
                    non_na = s_try.dropna()
                    has_frac = (non_na % 1 != 0).any()
                    out[name] = (
                        s_try.astype("Float64") if has_frac else s_try.astype("Int64")
                    )
                    continue
            out[name] = s
        else:
            out[name] = pd.Series(col)

    return pd.DataFrame(out)


# -----------------------
# Writer helpers (fixed-bytes encoders + sanitizers)
# -----------------------
def _to_fixed_bytes(
    series: pd.Series,
    safety_cap: int = 1 << 16,
    *,
    encoding: str = "utf-8",
    errors: str = "replace",
) -> pd.Series:
    """Convert string-like series to fixed-width NumPy bytes (|S{N})."""

    def _to_text(x):
        if isinstance(x, (dict, list, tuple, set)):
            return json.dumps(x, ensure_ascii=False)
        if x is None:
            return ""
        if isinstance(x, float) and np.isnan(x):
            return ""
        return str(x)

    s = series
    if pd.api.types.is_object_dtype(s):
        s = s.map(_to_text)
    else:
        s = s.astype("string").fillna("").astype(str)

    if len(s):
        lens = s.apply(lambda v: len(v.encode(encoding, errors=errors)) if v else 0)
        max_len = int(lens.max() or 0)
    else:
        max_len = 0

    max_len = max(1, min(safety_cap, max_len))
    dtype_s = f"|S{max_len}"

    arr = np.fromiter(
        (v.encode(encoding, errors=errors) for v in s), dtype=dtype_s, count=len(s)
    )
    return pd.Series(arr, index=series.index)


def _bool_to_fixed_bytes(series: pd.Series, *, encoding: str) -> pd.Series:
    """Encode BooleanDtype/boolean as fixed-width bytes 'true'/'false'/''."""
    if not (
        pd.api.types.is_bool_dtype(series)
        or getattr(series.dtype, "name", "") == "BooleanDtype"
    ):
        return series

    s = series.astype("boolean")
    s_txt = s.map(lambda x: "true" if x is True else "false" if x is False else "")
    return _to_fixed_bytes(s_txt, encoding=encoding, errors="replace")


def _sanitize_for_hdf5(df: pd.DataFrame) -> pd.DataFrame:
    """Sanitize DataFrame for HDF5 writing (strings/booleans to bytes, stable ints)."""
    df = df.copy()

    # Drop all-missing columns
    all_missing = [c for c in df.columns if df[c].isna().all()]
    if all_missing:
        df.drop(columns=all_missing, inplace=True)

    # Avoid accidental boolean inference
    df = df.convert_dtypes(dtype_backend="numpy_nullable", convert_boolean=False)

    for c in df.columns:
        s = df[c]

        # Datetimes
        if pd.api.types.is_datetime64_any_dtype(s):
            df[c] = pd.to_datetime(s, errors="coerce").astype("datetime64[ns]")
            continue

        # Integers
        if pd.api.types.is_integer_dtype(s):
            if pd.api.types.is_extension_array_dtype(s):  # Int64/Int32/... with <NA>
                if s.isna().any():
                    df[c] = s.astype("float64")
                else:
                    name = s.dtype.name.lower()
                    if name.endswith("8"):
                        df[c] = s.astype("int16")  # avoid 8-bit pitfalls
                    elif name.endswith("16"):
                        df[c] = s.astype("int16")
                    elif name.endswith("32"):
                        df[c] = s.astype("int32")
                    else:
                        df[c] = s.astype("int64")
            elif str(s.dtype).lower().endswith("8"):
                df[c] = s.astype("int16")
            continue

        # Booleans (strict)
        if (
            pd.api.types.is_bool_dtype(s)
            or getattr(s.dtype, "name", "") == "BooleanDtype"
        ):
            df[c] = _bool_to_fixed_bytes(s, encoding="utf-8")
            continue

        # Categories
        if pd.api.types.is_categorical_dtype(s):
            df[c] = s.cat.codes.astype("int32")
            continue

        # Floats ok
        if pd.api.types.is_float_dtype(s):
            continue

        # Strings/objects/arrow-backed strings -> bytes
        dtype_name = getattr(s.dtype, "name", str(s.dtype))
        if (
            pd.api.types.is_string_dtype(s)
            or pd.api.types.is_object_dtype(s)
            or ("string" in dtype_name)
        ):
            df[c] = _to_fixed_bytes(s, encoding="utf-8", errors="replace")
            continue

        if df[c].dtype == "object":
            df[c] = _to_fixed_bytes(s, encoding="utf-8", errors="replace")

    bad = [c for c in df.columns if df[c].dtype == "object"]
    if bad:
        raise TypeError(
            f"HDF5 sanitizer: unsupported object dtypes after conversion: {bad}"
        )

    return df


def _sanitize_for_fits(df: pd.DataFrame) -> pd.DataFrame:
    """Sanitize DataFrame for FITS writing (strings/booleans to ASCII bytes; promote 8-bit ints)."""
    df = df.copy()

    # Drop all-missing columns
    all_missing = [c for c in df.columns if df[c].isna().all()]
    if all_missing:
        df.drop(columns=all_missing, inplace=True)

    # Avoid accidental boolean inference
    df = df.convert_dtypes(dtype_backend="numpy_nullable", convert_boolean=False)

    for c in df.columns:
        s = df[c]

        # Datetimes
        if pd.api.types.is_datetime64_any_dtype(s):
            df[c] = pd.to_datetime(s, errors="coerce").astype("datetime64[ns]")
            continue

        # Integers
        if pd.api.types.is_integer_dtype(s):
            if pd.api.types.is_extension_array_dtype(s):
                if s.isna().any():
                    df[c] = s.astype("float64")
                else:
                    name = s.dtype.name.lower()
                    if name.endswith("8"):
                        df[c] = s.astype(
                            "int16"
                        )  # promote 8-bit -> 16-bit (avoid FITS LOGICAL)
                    elif name.endswith("16"):
                        df[c] = s.astype("int16")
                    elif name.endswith("32"):
                        df[c] = s.astype("int32")
                    else:
                        df[c] = s.astype("int64")
            else:
                npname = str(s.dtype).lower()
                if npname.endswith("8"):
                    df[c] = s.astype("int16")  # promote 8-bit -> 16-bit
            continue

        # Booleans (strict)
        if (
            pd.api.types.is_bool_dtype(s)
            or getattr(s.dtype, "name", "") == "BooleanDtype"
        ):
            df[c] = _bool_to_fixed_bytes(s, encoding="ascii")
            continue

        # Categories
        if pd.api.types.is_categorical_dtype(s):
            df[c] = s.cat.codes.astype("int32")
            continue

        # Floats ok
        if pd.api.types.is_float_dtype(s):
            continue

        # Strings/objects/pyarrow strings -> ASCII bytes
        dtype_name = getattr(s.dtype, "name", str(s.dtype))
        if (
            pd.api.types.is_string_dtype(s)
            or pd.api.types.is_object_dtype(s)
            or ("string" in dtype_name)
        ):
            df[c] = _to_fixed_bytes(s, encoding="ascii", errors="replace")
            continue

        if df[c].dtype == "object":
            df[c] = _to_fixed_bytes(s, encoding="ascii", errors="replace")

    bad = [c for c in df.columns if df[c].dtype == "object"]
    if bad:
        raise TypeError(
            f"FITS sanitizer: unsupported object dtypes after conversion: {bad}"
        )

    return df


# -----------------------
# Save API (Parquet/CSV/HDF5/FITS/HATS) + placement of writer helpers
# -----------------------
def save_dataframe(
    df,
    output_path,
    format_,
    *,
    temp_dir=None,
    client=None,
    logger=None,
    hats_size_threshold_mb=200,
):
    """Save DataFrame to disk in the requested format.

    Parquet: write via PyArrow directly.
    CSV/HDF5/FITS: sanitize dtypes for compatibility.
    HATS: write LSDB catalog directory.

    Args:
        df: Input pandas or Dask DataFrame.
        output_path: Path without extension.
        format_: Output format (parquet, csv, hdf5, fits, hats).
        temp_dir: Temporary directory used to stage parquet for large HATS outputs.
        client: Dask client used by hats-import for large HATS outputs.
        logger: Logger-like object.
        hats_size_threshold_mb: Max in-memory fast-path size for HATS output.

    Raises:
        ValueError: If the output format is unsupported.
    """
    ext = format_.lower()

    if ext == "parquet":
        try:
            df2 = df.reset_index(drop=True)
        except Exception:
            df2 = df
        table = pa.Table.from_pandas(df2, preserve_index=False)
        pq.write_table(table, f"{output_path}.parquet")
        return

    if ext == "hdf5":
        # Strict: integers remain integers; booleans only if dtype is boolean
        df_h5 = _sanitize_for_hdf5(df)
        tables_io.write(df_h5, f"{output_path}.hdf5")
        return

    if ext == "hats":
        _write_hats_output(
            df,
            f"{output_path}.hats",
            ra_column="ra",
            dec_column="dec",
            temp_dir=temp_dir,
            client=client,
            logger=logger,
            size_threshold_mb=hats_size_threshold_mb,
        )
        return

    # CSV/FITS start from numpy_nullable to stabilize null semantics
    df_np = df.convert_dtypes(dtype_backend="numpy_nullable", convert_boolean=False)

    if ext == "csv":
        df_np.to_csv(f"{output_path}.csv", index=False)
        return

    if ext == "fits":
        # Promote 8-bit ints to int16; write strings/booleans as bytes/text safely
        df_fits = _sanitize_for_fits(df_np).reset_index(drop=True)
        table = Table.from_pandas(df_fits)
        table.write(f"{output_path}.fits", overwrite=True)
        return

    raise ValueError(f"Unsupported output format: {format_}")


def _write_hats_output(
    data,
    output_dir: str,
    ra_column: str = "ra",
    dec_column: str = "dec",
    *,
    temp_dir=None,
    client=None,
    logger=None,
    size_threshold_mb: int = 200,
):
    """Persist output dataframe as a HATS catalog directory."""
    output_path = Path(output_dir)

    if output_path.exists():
        if output_path.is_dir():
            shutil.rmtree(output_path, ignore_errors=True)
        else:
            output_path.unlink(missing_ok=True)

    if not hasattr(data, "columns") or not {ra_column, dec_column}.issubset(set(data.columns)):
        raise ValueError(
            f"HATS output requires '{ra_column}' and '{dec_column}' columns in the final data."
        )

    if _is_dask_dataframe(data):
        _log_info(
            logger,
            "Writing lazy HATS output via parquet staging before threshold-based HATS build: %s",
            output_path,
        )
        parquet_path = _stage_hats_output_parquet(data, output_path, temp_dir, logger)
        build_collection_with_retry(
            parquet_path=parquet_path,
            output_path=output_path,
            output_artifact_name=output_path.name,
            catalog_artifact_name="catalog",
            margin_artifact_name=_margin_artifact_name(),
            client=client,
            logger=logger,
            ra_column=ra_column,
            dec_column=dec_column,
            try_margin=True,
            size_threshold_mb=size_threshold_mb,
        )
        return

    data_size_mb = _dataframe_memory_mb(data)
    if data_size_mb > size_threshold_mb:
        _log_info(
            logger,
            "HATS output dataframe is %.1f MB; using parquet staging and hats-import: %s",
            data_size_mb,
            output_path,
        )
        parquet_path = _stage_hats_output_parquet(data, output_path, temp_dir, logger)
        build_collection_with_retry(
            parquet_path=parquet_path,
            output_path=output_path,
            output_artifact_name=output_path.name,
            catalog_artifact_name="catalog",
            margin_artifact_name=_margin_artifact_name(),
            client=client,
            logger=logger,
            ra_column=ra_column,
            dec_column=dec_column,
            try_margin=True,
            size_threshold_mb=0,
        )
        return

    try:
        import lsdb
    except Exception as error:
        raise RuntimeError(
            "lsdb is required to export HATS output in combine_redshift_dedup."
        ) from error

    try:
        data = data.convert_dtypes(dtype_backend="numpy_nullable", convert_boolean=False)
    except Exception:
        pass

    _log_info(
        logger,
        "HATS output dataframe is %.1f MB; using in-memory LSDB fast path: %s",
        data_size_mb,
        output_path,
    )
    catalog = lsdb.from_dataframe(
        data,
        catalog_name="catalog",
        use_pyarrow_types=False,
        ra_column=ra_column,
        dec_column=dec_column,
    )
    _write_catalog_like_to_hats(catalog, output_path)
    _normalize_collection_margin_name(
        output_path,
        old_margin_name="catalog_5arcs",
        new_margin_name=_margin_artifact_name(),
    )


def _is_dask_dataframe(data):
    return isinstance(data, dd.DataFrame) or (
        hasattr(data, "to_parquet") and hasattr(data, "map_partitions")
    )


def _dataframe_memory_mb(data):
    try:
        return float(data.memory_usage(deep=True).sum()) / 1024**2
    except Exception:
        return float("inf")


def _rename_duplicate_columns_pd(pdf: pd.DataFrame, logger) -> pd.DataFrame:
    """Make pandas columns unique by appending __dupN."""
    cols = pd.Index(map(str, pdf.columns))
    if not cols.has_duplicates:
        return pdf

    seen: dict[str, int] = {}
    new_cols: list[str] = []
    renamed: list[tuple[str, str]] = []

    for col in cols:
        if col not in seen:
            seen[col] = 0
            new_cols.append(col)
            continue

        seen[col] += 1
        new_col = f"{col}__dup{seen[col]}"
        while new_col in seen:
            seen[col] += 1
            new_col = f"{col}__dup{seen[col]}"
        seen[new_col] = 0
        new_cols.append(new_col)
        renamed.append((col, new_col))

    if renamed:
        sample = ", ".join([f"{old}->{new}" for old, new in renamed[:6]])
        _log_warning(
            logger,
            "Renamed duplicate columns (pandas): %s%s",
            sample,
            " ..." if len(renamed) > 6 else "",
        )

    out = pdf.copy()
    out.columns = new_cols
    return out


def _cast_hats_fast_path_columns(dfp: pd.DataFrame) -> pd.DataFrame:
    """Apply stable dtypes before the in-memory LSDB fast path."""
    for col, pd_dtype in [
        ("CRD_ID", _HATS_DTYPE_STR),
        ("id", _HATS_DTYPE_STR),
        ("source", _HATS_DTYPE_STR),
        ("survey", _HATS_DTYPE_STR),
        ("instrument_type", _HATS_DTYPE_STR),
        ("instrument_type_homogenized", _HATS_DTYPE_STR),
        ("ra", _HATS_DTYPE_FLOAT),
        ("dec", _HATS_DTYPE_FLOAT),
        ("z", _HATS_DTYPE_FLOAT),
        ("z_err", _HATS_DTYPE_FLOAT),
        ("z_flag", _HATS_DTYPE_FLOAT),
        ("z_flag_homogenized", _HATS_DTYPE_FLOAT),
        ("tie_result", _HATS_DTYPE_INT8),
        ("is_in_DP1_fields", _HATS_DTYPE_INT),
        ("is_in_rubin_footprint", _HATS_DTYPE_INT),
        ("compared_to", _HATS_DTYPE_STR),
    ]:
        if col in dfp.columns:
            try:
                dfp[col] = dfp[col].astype(pd_dtype)
            except Exception:
                pass

    for col in map(str, dfp.columns):
        if col.startswith("CRD_ID_prev") or col.startswith("compared_to_prev"):
            try:
                dfp[col] = dfp[col].astype(_HATS_DTYPE_STR)
            except Exception:
                pass

    return dfp


def _margin_artifact_name(margin_threshold: float = 5.0) -> str:
    threshold = int(margin_threshold) if float(margin_threshold).is_integer() else margin_threshold
    return f"margin_{threshold}arcs"


def _normalize_collection_margin_name(
    collection_path: Path,
    old_margin_name: str,
    new_margin_name: str,
):
    """Rename LSDB-generated margin directory and update collection metadata."""
    old_margin_path = collection_path / old_margin_name
    new_margin_path = collection_path / new_margin_name

    if old_margin_path.is_dir() and old_margin_path != new_margin_path:
        if new_margin_path.exists():
            shutil.rmtree(new_margin_path, ignore_errors=True)
        old_margin_path.rename(new_margin_path)

    properties_path = collection_path / "collection.properties"
    if not properties_path.exists():
        return

    text = properties_path.read_text(encoding="utf-8")
    text = text.replace(f"all_margins={old_margin_name}", f"all_margins={new_margin_name}")
    text = text.replace(f"default_margin={old_margin_name}", f"default_margin={new_margin_name}")
    properties_path.write_text(text, encoding="utf-8")


def _stage_hats_output_parquet(data, output_path: Path, temp_dir, logger):
    """Write final output as parquet parts for hats-import without full materialization."""
    staging_root = Path(temp_dir) if temp_dir else output_path.parent / "temp"
    parquet_path = staging_root / f"{output_path.stem}_hats_parquet"

    if parquet_path.exists():
        shutil.rmtree(parquet_path, ignore_errors=True)
    parquet_path.mkdir(parents=True, exist_ok=True)

    _log_info(logger, "Staging HATS output parquet at %s", parquet_path)
    if _is_dask_dataframe(data):
        data.to_parquet(
            str(parquet_path),
            engine="pyarrow",
            write_index=False,
            overwrite=True,
        )
    else:
        table = pa.Table.from_pandas(data.reset_index(drop=True), preserve_index=False)
        pq.write_table(table, parquet_path / "part-0.parquet")

    return parquet_path


def build_collection_with_retry(
    parquet_path,
    logs_dir=None,
    logger=None,
    client=None,
    try_margin: bool = True,
    *,
    schema_hints: dict | None = None,
    size_threshold_mb: int = 200,
    output_path=None,
    output_artifact_name: str | None = None,
    catalog_artifact_name: str | None = None,
    margin_artifact_name: str | None = None,
    margin_threshold: float = 5.0,
    ra_column: str = "ra",
    dec_column: str = "dec",
) -> str:
    """Build a HATS collection from a Parquet folder.

    Small parquet inputs are loaded in memory and written with LSDB. Larger
    inputs use hats-import, which can run through the provided Dask client.
    """
    del logs_dir, schema_hints  # Kept for API compatibility with specz callers.

    parquet_path = Path(parquet_path).resolve()
    if output_path is not None:
        collection_path = Path(output_path).resolve()
        parent_dir = collection_path.parent
        collection_artifact_name = output_artifact_name or collection_path.name
        catalog_artifact_name = catalog_artifact_name or collection_path.stem
    else:
        parent_dir = parquet_path.parent
        collection_artifact_name = output_artifact_name or f"{parquet_path.name}_hats"
        catalog_artifact_name = catalog_artifact_name or collection_artifact_name
        collection_path = parent_dir / collection_artifact_name
    margin_artifact_name = margin_artifact_name or f"{catalog_artifact_name}_{int(margin_threshold)}arcs"

    base_path = parquet_path / "base"
    pattern = str(base_path / "*.parquet") if base_path.exists() else str(parquet_path / "*.parquet")
    in_file_paths = sorted(glob.glob(pattern))
    if not in_file_paths:
        raise ValueError(f"No Parquet files found at '{parquet_path}'")

    try:
        total_size_mb = sum(os.path.getsize(path) for path in in_file_paths) / 1024**2
    except Exception:
        total_size_mb = float("inf")

    if total_size_mb <= size_threshold_mb:
        try:
            import lsdb

            _log_info(
                logger,
                "Small catalog (%.1f MB). Building collection via fast path -> %s",
                total_size_mb,
                collection_path,
            )
            pdf_list = [pd.read_parquet(path) for path in in_file_paths]
            dfp = (
                pd.concat(pdf_list, ignore_index=True)
                if len(pdf_list) > 1
                else pdf_list[0]
            )
            dfp = _rename_duplicate_columns_pd(dfp, logger)
            dfp = _cast_hats_fast_path_columns(dfp)

            catalog = lsdb.from_dataframe(
                dfp,
                catalog_name=catalog_artifact_name,
                ra_column=ra_column,
                dec_column=dec_column,
                use_pyarrow_types=True,
            )
            catalog.write_catalog(str(collection_path), as_collection=True, overwrite=True)
            _log_info(logger, "Finished collection fast-path: %s", collection_path)
            return str(collection_path)
        except Exception as error:
            _log_warning(
                logger,
                "Collection fast-path failed for %s (%s: %s). Falling back to hats_import.",
                collection_artifact_name,
                type(error).__name__,
                error,
            )

    try:
        from hats_import import CollectionArguments
        from hats_import.collection.run_import import run as run_collection_import
    except Exception as error:
        raise RuntimeError(
            "hats-import is required to export large HATS output in combine_redshift_dedup."
        ) from error

    def _clean_partial():
        try:
            if collection_path.is_dir():
                shutil.rmtree(collection_path, ignore_errors=True)
        except Exception as error:
            _log_warning(logger, "Failed to remove partial '%s': %s", collection_path, error)

    _clean_partial()

    def _make_args(with_margin: bool):
        args = CollectionArguments(
            output_artifact_name=collection_artifact_name,
            output_path=str(parent_dir),
            resume=False,
        ).catalog(
            input_file_list=in_file_paths,
            file_reader="parquet",
            output_artifact_name=catalog_artifact_name,
            ra_column=ra_column,
            dec_column=dec_column,
        )
        if with_margin:
            args = args.add_margin(
                margin_threshold=margin_threshold,
                output_artifact_name=margin_artifact_name,
                is_default=True,
            )
        return args

    if try_margin:
        try:
            _log_info(
                logger,
                "Building collection WITH margin (import pipeline): %s",
                collection_artifact_name,
            )
            run_collection_import(_make_args(with_margin=True), client)
            return str(collection_path)
        except Exception as error:
            _log_warning(logger, "WITH margin failed: %s. Retrying WITHOUT margin...", error)
            _clean_partial()

    try:
        _log_info(
            logger,
            "Building collection WITHOUT margin (import pipeline): %s",
            collection_artifact_name,
        )
        run_collection_import(_make_args(with_margin=False), client)
    except Exception as error:
        _clean_partial()
        raise RuntimeError(f"Failed to build collection '{collection_artifact_name}': {error}") from error

    if not collection_path.is_dir():
        raise RuntimeError(f"hats-import did not create expected HATS output: {collection_path}")

    return str(collection_path)


def _log_info(logger, message, *args):
    if logger is not None:
        try:
            logger.info(message, *args)
            return
        except Exception:
            pass


def _log_warning(logger, message, *args):
    if logger is not None:
        try:
            logger.warning(message, *args)
            return
        except Exception:
            pass


def _write_catalog_like_to_hats(catalog_like, output_path: Path):
    """Write an LSDB catalog-like object as HATS."""
    write_catalog = getattr(catalog_like, "write_catalog", None)
    to_hats = getattr(catalog_like, "to_hats", None)

    attempts = []
    if callable(write_catalog):
        attempts.extend(
            [
                lambda: write_catalog(str(output_path), as_collection=True, overwrite=True),
                lambda: write_catalog(str(output_path), overwrite=True),
                lambda: write_catalog(str(output_path)),
            ]
        )
    if callable(to_hats):
        attempts.extend(
            [
                lambda: to_hats(
                    str(output_path),
                    catalog_name=output_path.stem,
                    overwrite=True,
                    progress_bar=False,
                    as_collection=False,
                ),
                lambda: to_hats(
                    str(output_path),
                    catalog_name=output_path.stem,
                    overwrite=True,
                    progress_bar=False,
                ),
                lambda: to_hats(str(output_path), overwrite=True, progress_bar=False),
                lambda: to_hats(str(output_path)),
            ]
        )

    if not attempts:
        raise RuntimeError("Could not find a compatible lsdb HATS writer in this environment.")

    last_error = None
    for run in attempts:
        try:
            result = run()
            if hasattr(result, "compute") and callable(result.compute):
                result.compute()
            return
        except TypeError as error:
            last_error = error
            continue
        except Exception as error:
            last_error = error
            continue

    raise RuntimeError(
        f"Could not write HATS output with available lsdb writer method: {last_error}"
    )
