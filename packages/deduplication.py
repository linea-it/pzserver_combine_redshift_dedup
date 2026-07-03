from __future__ import annotations

"""Deduplication for Combine Redshift Catalogs (CRC).

Builds a graph from `CRD_ID <-> compared_to` and resolves duplicates with
configurable priorities. Provides a pandas solver (`deduplicate_pandas`) and a
Dask/LSDB per-partition driver (`run_dedup_with_lsdb_map_partitions`).
"""

# -----------------------
# Standard library
# -----------------------
from typing import (
    Iterable,
    Mapping,
    Sequence,
    Dict,
    List,
)
import math
import logging
import hashlib

# -----------------------
# Third-party
# -----------------------
import numpy as np
import pandas as pd
import dask.dataframe as dd

# -----------------------
# Project
# -----------------------
from utils import get_phase_logger, log_phase

from lsdb.dask.merge_catalog_functions import (
    concat_align_catalogs,
    get_aligned_pixels_from_alignment,
    align_and_apply,
)

# -----------------------
# Logger (child of 'crc')
# -----------------------
LOGGER_NAME = "crc.dedup"


def _base_logger() -> logging.Logger:
    """Return the child base logger ('crc.dedup').

    Returns:
        logging.Logger: Base logger.
    """
    lg = logging.getLogger(LOGGER_NAME)
    lg.setLevel(logging.NOTSET)
    lg.propagate = True
    return lg


def _phase_logger() -> logging.LoggerAdapter:
    """Return a phase-aware logger (phase='deduplication').

    Returns:
        logging.LoggerAdapter: Logger with phase context.
    """
    return get_phase_logger("deduplication", _base_logger())


# -----------------------
# Public API
# -----------------------
__all__ = [
    "deduplicate_pandas",
    "run_dedup_with_lsdb_map_partitions",
    "count_global_edge_group_mismatches",
    "count_global_tie_invariant_violations",
]


# -----------------------
# Small helpers: string/parse/score
# -----------------------
def _canon_id_series(s: pd.Series) -> pd.Series:
    """Return canonical CRD-like IDs (strip and remove zero-width chars)."""
    t = s.astype("string")
    t = t.str.replace(r"[\u200B-\u200D\uFEFF]", "", regex=True)
    return t.str.strip()


def _norm_str(x) -> str | None:
    """Return a normalized string or None."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    s = str(x).strip()
    return s if s else None


def _canonical_group_id(value: object) -> np.int64:
    """Return a stable signed-63-bit ID for a canonical component key."""
    digest = hashlib.blake2b(str(value).encode("utf-8"), digest_size=8).digest()
    return np.int64(int.from_bytes(digest, "big") & ((1 << 63) - 1))


def validate_spatial_safety(
    crossmatch_radius_arcsec: float,
    margin_threshold_arcsec: float,
    margin_warning_fraction: float,
) -> None:
    """Validate parameters required by partition-local spatial deduplication."""
    if crossmatch_radius_arcsec <= 0.0:
        raise ValueError("crossmatch_radius_arcsec must be positive")
    if margin_threshold_arcsec <= 0.0:
        raise ValueError("margin_threshold_arcsec must be positive")
    if not 0.0 < margin_warning_fraction <= 1.0:
        raise ValueError("margin_warning_fraction must be in the interval (0, 1]")
    if crossmatch_radius_arcsec >= margin_threshold_arcsec:
        raise ValueError(
            "crossmatch_radius_arcsec must be smaller than "
            "margin_threshold_arcsec for partition-local deduplication "
            f"(radius={crossmatch_radius_arcsec}, margin={margin_threshold_arcsec})"
        )


def _assign_canonical_group_ids(
    df: pd.DataFrame, local_group_col: str, crd_col: str
) -> pd.Series:
    """Identify a component by a stable hash of its smallest CRD_ID."""
    canonical_key = (
        _canon_id_series(df[crd_col]).groupby(df[local_group_col]).transform("min")
    )
    return canonical_key.map(_canonical_group_id).astype("Int64")


def _validate_local_tie_invariants(
    df: pd.DataFrame,
    *,
    group_col: str,
    tie_col: str,
    z_flag_col: str = "z_flag_homogenized",
) -> None:
    """Validate winner/hard-tie semantics for every local component."""
    zf = pd.to_numeric(df[z_flag_col], errors="coerce")
    tie = pd.to_numeric(df[tie_col], errors="coerce")
    stars = zf.eq(6.0)
    if not tie[stars].eq(3).all():
        raise RuntimeError(
            "Local tie invariant failed: every star must have tie_result=3"
        )

    nonstars = df.loc[~stars, [group_col]].copy()
    nonstars["__tie"] = tie.loc[~stars].to_numpy()
    for gid_value, component in nonstars.groupby(group_col, dropna=False):
        values = component["__tie"]
        n_one = int(values.eq(1).sum())
        n_two = int(values.eq(2).sum())
        n_other_survivors = int((~values.isin([0, 1, 2])).sum())
        valid_single = n_one == 1 and n_two == 0
        valid_hard = n_one == 0 and n_two >= 2
        if n_other_survivors or not (valid_single or valid_hard):
            raise RuntimeError(
                "Local tie invariant failed for group "
                f"{gid_value}: tie_result counts={values.value_counts().to_dict()}"
            )


def _parse_compared_to_cell(val) -> List[str]:
    """Parse a single `compared_to` cell into a list of CRD_ID strings."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return []
    s = str(val).strip()
    if not s:
        return []
    return [t for t in (tok.strip() for tok in s.split(",")) if t]


def _score_instrument_type(
    series: pd.Series, priority_map: Mapping[str, int]
) -> pd.Series:
    """Score instrument types using a priority map."""
    norm_map = {str(k).strip().lower(): int(v) for k, v in priority_map.items()}

    def _score_one(v):
        v = _norm_str(v)
        return norm_map.get(v, 0) if v is not None else 0

    return series.map(_score_one).astype("int64")


def filter_pandas_by_tie_treatment(
    df: pd.DataFrame,
    option: str,
    *,
    tie_col: str = "tie_result",
    group_col: str = "group_id",
    random_state: int | None = None,
) -> tuple[pd.DataFrame, str, int]:
    """Apply final winner/hard-tie policy to an in-memory result frame.

    Returns the filtered frame, the effective option, and the number of
    hard-tie groups resolved by ``draw_one``.
    """
    if tie_col not in df.columns:
        raise RuntimeError(f"Expected '{tie_col}' column for remove-duplicates mode")

    effective = str(option or "remove_all").strip().lower()
    if effective not in {"remove_all", "keep_all", "draw_one"}:
        effective = "remove_all"

    tie_num = pd.to_numeric(df[tie_col], errors="coerce").fillna(0).astype("int8")
    resolved_groups = 0
    out = df

    if effective == "draw_one":
        if group_col not in df.columns:
            effective = "remove_all"
        else:
            gid = df[group_col]
            has_1 = tie_num.eq(1).groupby(gid, dropna=True).transform("any").fillna(False)
            has_2 = tie_num.eq(2).groupby(gid, dropna=True).transform("any").fillna(False)
            candidates = (~has_1) & has_2 & tie_num.eq(2) & gid.notna()
            candidate_positions = np.flatnonzero(
                candidates.to_numpy(dtype=bool, na_value=False)
            )
            if candidate_positions.size:
                rng = np.random.default_rng(random_state)
                winners = []
                candidate_groups = gid.iloc[candidate_positions]
                for _, positions in pd.Series(
                    candidate_positions, index=candidate_groups.to_numpy()
                ).groupby(level=0, sort=False):
                    winners.append(int(rng.choice(positions.to_numpy())))
                resolved_groups = len(winners)
                tie_num = tie_num.copy()
                tie_num.iloc[candidate_positions] = 0
                tie_num.iloc[winners] = 1
                out = df.assign(**{tie_col: tie_num})

    keep_mask = tie_num.isin([1, 2]) if effective == "keep_all" else tie_num.eq(1)
    return out.loc[keep_mask].copy(), effective, resolved_groups


# -----------------------
# Graph building
# -----------------------
def _split_cmp_vectorized(s: pd.Series) -> pd.Series:
    """Split `compared_to` values vectorially into lists."""
    s = s.astype("string")
    return s.str.split(",")


def _edge_rows_partition(
    part: pd.DataFrame,
    crd_col: str,
    compared_col: str,
    z_flag_col: str,
) -> pd.DataFrame:
    """Extract directed non-star graph edges from one dataframe partition."""
    zf = pd.to_numeric(part[z_flag_col], errors="coerce")
    work = part.loc[~zf.eq(6.0), [crd_col, compared_col]].copy()
    if work.empty:
        return pd.DataFrame(
            {
                "u": pd.Series(dtype="string[pyarrow]"),
                "v": pd.Series(dtype="string[pyarrow]"),
            }
        )
    work["u"] = _canon_id_series(work[crd_col])
    work["v"] = work[compared_col].astype("string").str.split(",")
    edges = work[["u", "v"]].explode("v", ignore_index=True)
    edges["v"] = _canon_id_series(edges["v"])
    edges = edges.dropna().loc[lambda frame: frame["u"].ne(frame["v"])]
    u = edges["u"].astype(str).to_numpy()
    v = edges["v"].astype(str).to_numpy()
    edges = pd.DataFrame({"u": np.minimum(u, v), "v": np.maximum(u, v)})
    return edges.drop_duplicates().astype(
        {"u": "string[pyarrow]", "v": "string[pyarrow]"}
    )


def count_global_edge_group_mismatches(
    df: dd.DataFrame,
    *,
    crd_col: str = "CRD_ID",
    compared_col: str = "compared_to",
    z_flag_col: str = "z_flag_homogenized",
    group_col: str = "group_id",
):
    """Return lazy counts of cross-group edges and dangling non-star edges."""
    meta = pd.DataFrame(
        {
            "u": pd.Series(dtype="string[pyarrow]"),
            "v": pd.Series(dtype="string[pyarrow]"),
        }
    )
    edges = df.map_partitions(
        _edge_rows_partition,
        crd_col,
        compared_col,
        z_flag_col,
        meta=meta,
    ).drop_duplicates()

    zf = dd.to_numeric(df[z_flag_col], errors="coerce")
    groups = df[[crd_col, group_col]].assign(is_star=zf.eq(6.0)).rename(
        columns={crd_col: "node", group_col: "node_group"}
    )
    groups = groups.assign(node=groups["node"].astype("string[pyarrow]"))
    groups = groups.drop_duplicates(subset=["node"])

    # Rename each lookup before merging and use same-name keys. Dask 2025.3
    # can lose a left_on/right_on key while lowering consecutive merge/rename
    # expressions, producing a spurious merge key of None.
    groups_u = groups.rename(
        columns={"node": "u", "node_group": "group_u", "is_star": "is_star_u"}
    )
    groups_v = groups.rename(
        columns={"node": "v", "node_group": "group_v", "is_star": "is_star_v"}
    )
    checked = edges.merge(groups_u, on="u", how="left")
    checked = checked.merge(groups_v, on="v", how="left")

    dangling = checked["group_u"].isna() | checked["group_v"].isna()
    both_nonstar = checked["is_star_u"].eq(False) & checked["is_star_v"].eq(False)
    mismatch = (
        (~dangling) & both_nonstar & checked["group_u"].ne(checked["group_v"])
    )
    return mismatch.sum(), dangling.sum()


def count_global_tie_invariant_violations(
    df: dd.DataFrame,
    *,
    group_col: str = "group_id",
    tie_col: str = "tie_result",
    z_flag_col: str = "z_flag_homogenized",
):
    """Return a lazy count of groups violating final tie-result semantics."""
    zf = dd.to_numeric(df[z_flag_col], errors="coerce")
    tie = dd.to_numeric(df[tie_col], errors="coerce")
    nonstars = df.loc[~zf.eq(6.0), [group_col]].assign(
        n=1,
        n0=tie.eq(0).astype("int8"),
        n1=tie.eq(1).astype("int8"),
        n2=tie.eq(2).astype("int8"),
        n_invalid=(~tie.isin([0, 1, 2])).astype("int8"),
    )
    stats = nonstars.groupby(group_col).agg(
        {"n": "sum", "n0": "sum", "n1": "sum", "n2": "sum", "n_invalid": "sum"}
    )
    valid_single = (
        stats["n1"].eq(1)
        & stats["n2"].eq(0)
        & stats["n0"].eq(stats["n"] - 1)
    )
    valid_hard = (
        stats["n1"].eq(0)
        & stats["n2"].ge(2)
        & stats["n0"].eq(stats["n"] - stats["n2"])
    )
    invalid = stats["n_invalid"].gt(0) | ~(valid_single | valid_hard)
    missing_group_rows = nonstars[group_col].isna().sum()
    return invalid.sum() + missing_group_rows


def _build_edges_fast(
    df: pd.DataFrame,
    *,
    crd_col: str,
    compared_col: str,
    zf_series: pd.Series | None = None,
    edge_log: bool = False,
):
    """Build undirected edges among NON-STAR rows (vectorized path).

    Returns:
        (nodes_index, edges_uv, diag) where:
          - nodes_index: node labels for the condensed graph
          - edges_uv: (E, 2) int32 array of undirected edges (u < v)
          - diag: dict with basic diagnostics (counts). If `edge_log` is False,
                  only cheap counts are filled; expensive ones are set to None.
    """
    # --- filter A-side (rows) to non-stars
    non_star_mask = pd.Series(True, index=df.index)
    if zf_series is not None:
        non_star_mask &= ~pd.to_numeric(zf_series, errors="coerce").eq(6)

    A_df = df.loc[non_star_mask, [crd_col, compared_col]].copy().reset_index(drop=True)
    if A_df.empty:
        diag = {
            "edge_log_enabled": bool(edge_log),
            "n_rows_nonstar": int(non_star_mask.sum()),
            "n_rows_star_excluded": int((~non_star_mask).sum()),
            "n_edges_raw": 0,
            "n_edges_kept": 0,
            "n_edges_starB_excluded": None if not edge_log else 0,
        }
        return pd.Index([], dtype="object"), np.empty((0, 2), dtype=np.int32), diag

    # Non-star IDs present on A-side
    present_nonstar = set(
        _canon_id_series(df.loc[non_star_mask, crd_col]).dropna().unique()
    )

    # If diagnostics are enabled, precompute star IDs (for B-side exclusion count)
    if edge_log and (zf_series is not None):
        star_ids = set(
            _canon_id_series(
                df.loc[pd.to_numeric(zf_series, errors="coerce").eq(6), crd_col]
            )
            .dropna()
            .unique()
        )
    else:
        star_ids = None  # skip expensive star-neighbor counting

    # explode compared_to
    cmp_lists = _split_cmp_vectorized(A_df[compared_col])
    edges_raw = (
        A_df[[crd_col]]
        .assign(v=cmp_lists)
        .rename(columns={crd_col: "u"})
        .explode("v", ignore_index=True)
    )

    # canonicalize ids
    edges_raw["u"] = _canon_id_series(edges_raw["u"])
    edges_raw["v"] = _canon_id_series(edges_raw["v"])

    # drop empties
    edges_raw = edges_raw.dropna(subset=["u", "v"])
    edges_raw = edges_raw[(edges_raw["u"] != "") & (edges_raw["v"] != "")]
    n_edges_raw = int(len(edges_raw))

    # Optional: count how many B-neighbors were stars (diagnostic only)
    if edge_log and (star_ids is not None) and n_edges_raw:
        n_edges_starB_excluded = int(edges_raw["v"].isin(star_ids).sum())
    else:
        n_edges_starB_excluded = None

    # keep only neighbors that are non-stars present on A-side
    edges_raw = edges_raw[edges_raw["v"].isin(present_nonstar)]
    if edges_raw.empty:
        diag = {
            "edge_log_enabled": bool(edge_log),
            "n_rows_nonstar": int(non_star_mask.sum()),
            "n_rows_star_excluded": int((~non_star_mask).sum()),
            "n_edges_raw": n_edges_raw,
            "n_edges_kept": 0,
            "n_edges_starB_excluded": n_edges_starB_excluded,
        }
        return pd.Index([], dtype="object"), np.empty((0, 2), dtype=np.int32), diag

    # unique undirected edges (u < v)
    nodes_edge = pd.Index(
        pd.unique(pd.concat([edges_raw["u"], edges_raw["v"]], ignore_index=True))
    )

    # --- EXTRA LOG (edge_log): sanity-check that no star IDs leaked into edge nodes
    if edge_log and (zf_series is not None) and len(nodes_edge):
        star_ids_fast = set(
            _canon_id_series(
                df.loc[pd.to_numeric(zf_series, errors="coerce").eq(6), crd_col]
            )
            .dropna()
            .unique()
        )
        leaked = [cid for cid in nodes_edge.astype("string") if cid in star_ids_fast]
        if leaked:
            _phase_logger().error(
                "Star IDs leaked into fast-path edge nodes (logic violation): "
                "count=%d sample=%s",
                len(leaked),
                leaked[:5],
            )

    id2ix = {cid: i for i, cid in enumerate(nodes_edge)}
    u = edges_raw["u"].map(id2ix).to_numpy(dtype=np.int32, copy=False)
    v = edges_raw["v"].map(id2ix).to_numpy(dtype=np.int32, copy=False)

    lo = np.minimum(u, v)
    hi = np.maximum(u, v)
    mask = lo != hi
    lo, hi = lo[mask], hi[mask]
    if lo.size == 0:
        diag = {
            "edge_log_enabled": bool(edge_log),
            "n_rows_nonstar": int(non_star_mask.sum()),
            "n_rows_star_excluded": int((~non_star_mask).sum()),
            "n_edges_raw": n_edges_raw,
            "n_edges_kept": 0,
            "n_edges_starB_excluded": n_edges_starB_excluded,
        }
        return nodes_edge, np.empty((0, 2), dtype=np.int32), diag

    uv = np.stack([lo, hi], axis=1)
    view = uv.view([("x", uv.dtype), ("y", uv.dtype)])
    uv = np.unique(view).view(uv.dtype).reshape(-1, 2)

    diag = {
        "edge_log_enabled": bool(edge_log),
        "n_rows_nonstar": int(non_star_mask.sum()),
        "n_rows_star_excluded": int((~non_star_mask).sum()),
        "n_edges_raw": n_edges_raw,
        "n_edges_kept": int(uv.shape[0]),
        # None when edge_log=False to indicate we skipped the costly check
        "n_edges_starB_excluded": n_edges_starB_excluded,
    }
    return nodes_edge, uv, diag


def _connected_components_scipy(n_nodes: int, edges_uv: np.ndarray) -> np.ndarray:
    """Compute connected components using SciPy."""
    if n_nodes == 0:
        return np.array([], dtype=np.int64)
    if edges_uv.size == 0:
        return np.arange(n_nodes, dtype=np.int64)

    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    data = np.ones(edges_uv.shape[0], dtype=np.int8)
    A = coo_matrix((data, (edges_uv[:, 0], edges_uv[:, 1])), shape=(n_nodes, n_nodes))
    A = A + A.T
    _, labels = connected_components(A, directed=False, return_labels=True)
    return labels


def _build_edges_pdf(df: pd.DataFrame, crd_col: str, compared_col: str) -> pd.DataFrame:
    """Build undirected edges (u, v) from `compared_to` (fallback)."""
    if df.empty:
        return pd.DataFrame(
            {"u": pd.Series([], dtype="string"), "v": pd.Series([], dtype="string")}
        )

    present_ids = set(df[crd_col].astype(str))
    a = df[crd_col].astype(str)
    b_lists = df[compared_col].apply(_parse_compared_to_cell)
    edges = (
        pd.DataFrame({"u": a, "v": b_lists})
        .explode("v", ignore_index=True)
        .dropna(subset=["v"])
    )
    if edges.empty:
        return pd.DataFrame(
            {"u": pd.Series([], dtype="string"), "v": pd.Series([], dtype="string")}
        )

    edges["v"] = edges["v"].astype(str)
    edges = edges[edges["v"].isin(present_ids)]
    u = edges["u"].astype(str)
    v = edges["v"].astype(str)
    lo = np.minimum(u.values, v.values)
    hi = np.maximum(u.values, v.values)
    mask = lo != hi
    if not mask.any():
        return pd.DataFrame(
            {"u": pd.Series([], dtype="string"), "v": pd.Series([], dtype="string")}
        )

    out = pd.DataFrame(
        {
            "u": pd.Series(lo[mask], dtype="string"),
            "v": pd.Series(hi[mask], dtype="string"),
        }
    ).drop_duplicates(ignore_index=True)
    return out


# -----------------------
# Dz clustering + CC fallback (DSU)
# -----------------------
def _collapse_within_dz(
    mask: pd.Series,
    gid: pd.Series,
    zvals: pd.Series,
    crd_s: pd.Series,
    threshold: float,
) -> pd.Series:
    """Keep a maximal low-to-high set separated by at least ``threshold``.

    Values strictly closer than the threshold to the last representative are
    redundant. Equality remains a hard-tie separation by design. Undefined
    redshifts lose whenever the same group has at least one defined redshift;
    all-undefined groups remain unresolved.
    """
    thr = float(threshold or 0.0)
    if thr <= 0.0:
        return mask

    m = mask.to_numpy(dtype=bool, na_value=False)
    if not m.any():
        return mask

    n = len(mask)
    pos_all = np.arange(n, dtype=np.int64)
    pos = pos_all[m]

    gid_arr = (
        pd.Index(gid).to_numpy() if isinstance(gid, pd.Series) else np.asarray(gid)
    )
    z_arr = pd.to_numeric(zvals, errors="coerce").to_numpy()
    crd_arr = crd_s.astype(str).to_numpy()

    # Missing redshift cannot remain a representative when the same component
    # contains at least one defined candidate. Preserve all-NaN components as
    # unresolved hard ties instead of choosing arbitrarily.
    candidate_frame = pd.DataFrame(
        {"pos": pos, "gid": gid_arr[pos], "z_defined": ~np.isnan(z_arr[pos])}
    )
    has_defined = candidate_frame.groupby("gid")["z_defined"].transform("any")
    drop_missing = has_defined.to_numpy() & ~candidate_frame["z_defined"].to_numpy()
    if drop_missing.any():
        m[candidate_frame.loc[drop_missing, "pos"].to_numpy(dtype=np.int64)] = False
        pos = pos[~drop_missing]

    pos_def = pos[~np.isnan(z_arr[pos])]
    if pos_def.size == 0:
        return mask

    order = np.lexsort((crd_arr[pos_def], z_arr[pos_def], gid_arr[pos_def]))
    pos_sorted = pos_def[order]
    winners = []
    last_gid = None
    last_z = None
    for current_pos in pos_sorted:
        current_gid = gid_arr[current_pos]
        current_z = float(z_arr[current_pos])
        if current_gid != last_gid or last_z is None or current_z - last_z >= thr:
            winners.append(int(current_pos))
            last_gid = current_gid
            last_z = current_z
    winners_pos = np.asarray(winners, dtype=np.int64)

    out_np = m.copy()
    out_np[pos_def] = False
    out_np[winners_pos] = True
    return pd.Series(out_np, index=mask.index)


class _DSU:
    """Disjoint Set Union for connected components."""

    __slots__ = ("p",)

    def __init__(self, n: int):
        """Initialize DSU with `n` elements."""
        self.p = list(range(n))

    def find(self, a: int) -> int:
        """Find set representative with path compression."""
        p = self.p
        while p[a] != a:
            p[a] = p[p[a]]
            a = p[a]
        return a

    def union(self, a: int, b: int) -> None:
        """Union sets of a and b (attach higher index under lower)."""
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            if ra < rb:
                self.p[rb] = ra
            else:
                self.p[ra] = rb


def _connected_components(
    nodes: Iterable[str],
    edges: pd.DataFrame,
) -> Dict[str, int]:
    """Compute connected components over nodes using undirected edges."""
    nodes = list(nodes)
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    dsu = _DSU(n)

    if not edges.empty:
        u = edges["u"].astype(str).values
        v = edges["v"].astype(str).values
        for uu, vv in zip(u, v):
            iu = idx.get(uu)
            iv = idx.get(vv)
            if iu is not None and iv is not None:
                dsu.union(iu, iv)

    roots = [dsu.find(i) for i in range(n)]
    root_to_gid: Dict[int, int] = {}
    next_gid = 0
    out: Dict[str, int] = {}
    for node, r in zip(nodes, roots):
        if r not in root_to_gid:
            root_to_gid[r] = next_gid
            next_gid += 1
        out[node] = root_to_gid[r]
    return out


# -----------------------
# 1-D safe helpers (numeric)
# -----------------------
def _series_1d_from(df_or_series) -> pd.Series:
    """Ensure a 1-D Series when a duplicated-name selection returns a DataFrame."""
    if isinstance(df_or_series, pd.DataFrame):
        return df_or_series.iloc[:, 0]
    return df_or_series


def _to_numeric(series_like) -> pd.Series:
    """Coerce to float (1-D safe, NaN for non-castable)."""
    s = _series_1d_from(series_like)
    return pd.to_numeric(s, errors="coerce")


# -----------------------
# Guard restore
# -----------------------
def _only_star_neighbors_series(col: pd.Series, star_ids: set[str]) -> pd.Series:
    """True when compared_to is non-empty AND all neighbors are star IDs."""
    s = col.astype("string").fillna("")
    lst = s.str.split(",")
    out = []
    for tokens in lst:
        toks = [t.strip() for t in tokens if t and t.strip()]
        out.append(bool(toks) and all((tok in star_ids) for tok in toks))
    return pd.Series(out, index=col.index, dtype="boolean")


def _apply_guard_restore_local(
    df: pd.DataFrame,
    *,
    crd_col: str,
    compared_col: str,
    zf_series: pd.Series | None,
    tie_col: str,
    tie_col_orig: str,
) -> pd.DataFrame:
    """Restore original tie_result for non-stars with empty/only-star neighbors.

    Args:
        df: Partition dataframe.
        crd_col: Name of the CRD_ID column.
        compared_col: Name of the compared_to column.
        zf_series: Optional z_flag series.
        tie_col: Name of the tie_result column.
        tie_col_orig: Name of the original tie_result column.

    Returns:
        pd.DataFrame: Updated dataframe.
    """
    if tie_col_orig not in df.columns:
        return df  # nothing to restore

    # Stars (fixed as 3) and empty compared_to mask.
    is_star = pd.Series(False, index=df.index)
    if zf_series is not None:
        is_star = pd.to_numeric(zf_series, errors="coerce").eq(6.0)

    cmp_str = df[compared_col].astype("string")
    cmp_empty = cmp_str.isna() | cmp_str.str.strip().eq("")

    # Local set of star IDs for this partition/view.
    star_ids = set(df.loc[is_star, crd_col].astype("string"))

    only_star_neighbors = (~cmp_empty) & _only_star_neighbors_series(
        df[compared_col], star_ids
    )

    # Rule:
    # - if star -> keep tie=3 (do not restore)
    # - if non-star and (empty compared_to OR only star neighbors) -> restore
    restore_mask = (~is_star) & (cmp_empty | only_star_neighbors)

    # Apply restoration.
    df.loc[restore_mask, tie_col] = df.loc[restore_mask, tie_col_orig]
    return df


# -----------------------
# Per-group resolver
# -----------------------
def _resolve_group(
    g: pd.DataFrame,
    *,
    crd_col: str,
    z_col: str,
    tiebreaking_priority: Sequence[str],
    instrument_type_priority: Mapping[str, int] | None,
    delta_z_threshold: float,
) -> pd.DataFrame:
    """Resolve ties within a single connected component."""
    crd = crd_col
    out = g[[crd]].copy()
    out["tie_result_new"] = 0

    star_mask = pd.Series(False, index=g.index)
    if "z_flag_homogenized" in g.columns:
        zf_series = _to_numeric(g["z_flag_homogenized"])
        star_mask = zf_series.eq(6)
        out.loc[star_mask.index[star_mask], "tie_result_new"] = 3

    cand = g[~star_mask].copy()
    if cand.empty:
        return out[[crd, "tie_result_new"]]

    survivors_pos = np.arange(len(cand), dtype=np.int64)

    for col in tiebreaking_priority:
        if survivors_pos.size <= 1:
            break

        sub = cand.iloc[survivors_pos]
        col_vals_1d = _series_1d_from(sub[col])

        if col == "instrument_type_homogenized":
            if instrument_type_priority is None:
                raise ValueError(
                    "instrument_type_priority is required when "
                    "'instrument_type_homogenized' is used in tiebreaking_priority."
                )
            scores = _score_instrument_type(col_vals_1d, instrument_type_priority)
        else:
            scores = _to_numeric(col_vals_1d)

        scores = scores.astype("float64")
        scores = scores.where(~scores.isna(), other=-np.inf)
        mx = scores.max()
        if np.isneginf(mx):
            continue

        keep_mask_np = scores.eq(mx).to_numpy()
        survivors_pos = survivors_pos[keep_mask_np]

    if survivors_pos.size > 1 and (delta_z_threshold or 0.0) > 0.0:
        sub = cand.iloc[survivors_pos]
        zvals = _to_numeric(_series_1d_from(sub[z_col])).dropna()
        if not zvals.empty and (zvals.max() - zvals.min()) <= float(delta_z_threshold):
            crds = sub[crd].astype(str).to_numpy()
            keep_local = np.argmin(crds)
            survivors_pos = np.array([survivors_pos[keep_local]], dtype=np.int64)

    if survivors_pos.size == 1:
        winner_label = cand.index[survivors_pos[0]]
        out.loc[winner_label, "tie_result_new"] = 1
    elif survivors_pos.size > 1:
        winner_labels = cand.index.take(survivors_pos)
        out.loc[winner_labels, "tie_result_new"] = 2

    return out[[crd, "tie_result_new"]]


# -----------------------
# Public API (Pandas)
# -----------------------
def deduplicate_pandas(
    df: pd.DataFrame,
    tiebreaking_priority: Sequence[str],
    instrument_type_priority: Mapping[str, int] | None = None,
    *,
    delta_z_threshold: float | int | None = 0.0,
    crd_col: str = "CRD_ID",
    compared_col: str = "compared_to",
    z_col: str = "z",
    tie_col: str = "tie_result",
    edge_log: bool = False,
    partition_tag: str | None = None,
    logger: logging.LoggerAdapter | None = None,
    group_col: str | None = None,  # new
) -> pd.DataFrame:
    """Graph-based deduplication with vectorized per-group resolution and Dz collapse.

    Args:
        df: Input dataframe.
        tiebreaking_priority: Ordered columns for tie-breaking.
        instrument_type_priority: Optional map for instrument type scoring.
        delta_z_threshold: Dz threshold for final disambiguation.
        crd_col: Name of the ID column.
        compared_col: Name of the neighbors column.
        z_col: Name of the redshift column.
        tie_col: Name of the output tie-result column.
        edge_log: Enable edge diagnostics during graph build.
        partition_tag: Optional tag to namespace logs.
        logger: Optional logger for diagnostics.
        group_col: If set, emit component/group id in this column.

    Returns:
        pd.DataFrame: Deduplicated dataframe with tie labels.
    """
    required = {crd_col, compared_col, z_col}
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    out = df.copy()

    tie_col_orig = f"{tie_col}_orig"
    if tie_col in out.columns:
        try:
            out[tie_col_orig] = out[tie_col].astype("Int8")
        except Exception:
            out[tie_col_orig] = out[tie_col]

    crd_norm = out[crd_col].astype("string").str.strip()
    priority_set = set(tiebreaking_priority)

    zf_series: pd.Series | None = None
    if "z_flag_homogenized" in out.columns:
        zf_series = _to_numeric(out["z_flag_homogenized"])

    # Pass edge_log down so diagnostics are computed only when requested.
    nodes_edge, edges_uv, diag = _build_edges_fast(
        out,
        crd_col=crd_col,
        compared_col=compared_col,
        zf_series=zf_series,
        edge_log=edge_log,
    )

    # Logging behavior (quiet for partitions).
    if edge_log:
        lg = logger or _phase_logger()
        tag = f"[{partition_tag}]" if partition_tag else "[global]"
        starB = diag.get("n_edges_starB_excluded")
        if isinstance(starB, int) and starB > 0:
            lg.warning(
                "%s Star neighbors excluded during edge build: star_rows_excl=%d, edges_raw=%d, starB_excl=%d, edges_kept=%d",
                tag,
                diag.get("n_rows_star_excluded", 0),
                diag.get("n_edges_raw", 0),
                starB,
                diag.get("n_edges_kept", 0),
            )
        elif partition_tag is None:
            lg.info(
                "%s Edge build summary: star_rows_excl=%d, edges_raw=%d, starB_excl=%s, edges_kept=%d",
                tag,
                diag.get("n_rows_star_excluded", 0),
                diag.get("n_edges_raw", 0),
                str(starB),
                diag.get("n_edges_kept", 0),
            )

    labels_edge = _connected_components_scipy(len(nodes_edge), edges_uv)

    if labels_edge.size:
        s_map_index = pd.Index(nodes_edge).astype("string").str.strip()
        s_map = pd.Series(labels_edge.astype("int64"), index=s_map_index)
        mapped = crd_norm.map(s_map)
    else:
        mapped = pd.Series(np.nan, index=out.index, dtype="float64")

    # Boolean NumPy mask of rows without a mapped group id from the fast path
    na_mask = mapped.isna().to_numpy()

    # Preallocate group ids array
    gids = np.full(len(out), -1, dtype=np.int64)

    # Fill groups for rows that already mapped via the fast path
    if (~na_mask).any():
        gids[~na_mask] = mapped[~na_mask].to_numpy(dtype=np.int64, copy=False)

    # Try to bridge NA rows via neighbor groups from the fast path label map
    if na_mask.any() and labels_edge.size:
        pos_na = np.flatnonzero(na_mask)
        cmp_str_all = out[compared_col].astype("string").str.strip()
        cmp_lists = cmp_str_all.iloc[pos_na].str.split(",")
        sub = pd.DataFrame({"pos": pos_na, "nbr": cmp_lists}).explode(
            "nbr", ignore_index=False
        )
        sub["nbr"] = sub["nbr"].astype("string").str.strip()
        sub = sub[(sub["nbr"].notna()) & (sub["nbr"] != "")]
        if not sub.empty:
            sub["nbr_gid"] = sub["nbr"].map(s_map)
            sub = sub[sub["nbr_gid"].notna()]
            if not sub.empty:
                bridged = (
                    sub.groupby("pos", sort=False)["nbr_gid"].min().astype("int64")
                )
                gids[bridged.index.to_numpy()] = bridged.to_numpy(
                    dtype=np.int64, copy=False
                )
                na_mask[bridged.index.to_numpy()] = False

    # Fallback for rows still without group id (avoid mixing stars in graph).
    if na_mask.any():
        pos_na = np.flatnonzero(na_mask)

        # Stars within NA.
        if zf_series is not None:
            is_star_na = np.asarray(
                zf_series.iloc[pos_na].eq(6).fillna(False), dtype=bool
            )
        else:
            is_star_na = np.zeros(pos_na.size, dtype=bool)

        pos_na_nonstar = pos_na[~is_star_na]

        # Next free ID, compatible with fast-path labels.
        next_gid = int(labels_edge.max()) + 1 if labels_edge.size else 0

        # NA non-star rows.
        if pos_na_nonstar.size:
            crd_arr_all = crd_norm.to_numpy()
            cmp_arr_all = out[compared_col].astype("string").str.strip().to_numpy()

            crd_arr = crd_arr_all[pos_na_nonstar]
            cmp_arr = cmp_arr_all[pos_na_nonstar]

            sub_df = pd.DataFrame({crd_col: crd_arr, compared_col: cmp_arr})
            edges_na = _build_edges_pdf(
                sub_df, crd_col=crd_col, compared_col=compared_col
            )
            nodes_na = list(pd.Index(sub_df[crd_col]).astype("string").unique())

            if nodes_na:
                if edges_na.empty:
                    # No edges: each row becomes a singleton.
                    gids[pos_na_nonstar] = np.arange(
                        next_gid, next_gid + len(pos_na_nonstar), dtype=np.int64
                    )
                    next_gid += len(pos_na_nonstar)
                else:
                    # With edges: real components.
                    comp_map = _connected_components(nodes_na, edges_na)
                    gids[pos_na_nonstar] = next_gid + np.fromiter(
                        (comp_map.get(str(cid), -1) for cid in crd_arr),
                        dtype=np.int64,
                        count=len(crd_arr),
                    )
                    next_gid += max(comp_map.values()) + 1 if comp_map else 0
            else:
                # No nodes: singleton per row.
                gids[pos_na_nonstar] = np.arange(
                    next_gid, next_gid + len(pos_na_nonstar), dtype=np.int64
                )
                next_gid += len(pos_na_nonstar)

            # Mark as mapped to avoid later collisions.
            na_mask[pos_na_nonstar] = False

        # Stars (each is a singleton).
        if is_star_na.any():
            n_star = int(is_star_na.sum())
            gids[pos_na[is_star_na]] = np.arange(
                next_gid, next_gid + n_star, dtype=np.int64
            )
            na_mask[pos_na[is_star_na]] = False
            next_gid += n_star

    out["__group__"] = gids
    gid = out["__group__"]
    crd_s = crd_norm

    group_sizes = pd.Series(1, index=out.index).groupby(gid).transform("sum")
    is_singleton = group_sizes.eq(1)

    if zf_series is None:
        is_star = pd.Series(False, index=out.index)
    else:
        is_star = zf_series.eq(6)

    is_singleton_np = is_singleton.to_numpy(dtype=bool, na_value=False)
    is_star_np = is_star.to_numpy(dtype=bool, na_value=False)

    tr = np.zeros(len(out), dtype=np.int8)
    tr[is_star_np] = 3
    tr[is_singleton_np & ~is_star_np] = 1

    is_multi = ~is_singleton
    non_star = ~is_star
    survivors = (is_multi & non_star).copy()

    zf_num = _to_numeric(
        out.get("z_flag_homogenized", pd.Series(np.nan, index=out.index))
    ).astype("float64")

    if "instrument_type_homogenized" in priority_set:
        if instrument_type_priority is None:
            raise ValueError(
                "instrument_type_priority is required when "
                "'instrument_type_homogenized' is used in tiebreaking_priority."
            )
        it_scores = _score_instrument_type(
            out["instrument_type_homogenized"], instrument_type_priority
        ).astype("float64")
    else:
        it_scores = pd.Series(0.0, index=out.index, dtype="float64")

    for col in tiebreaking_priority:
        if not survivors.to_numpy(dtype=bool, na_value=False).any():
            break

        if col == "instrument_type_homogenized":
            scores = it_scores
        elif col == "z_flag_homogenized":
            scores = zf_num
        else:
            scores = _to_numeric(out[col]).astype("float64")

        scores = scores.where(~scores.isna(), other=-np.inf)
        s_eff = scores.where(survivors, other=-np.inf)
        gmax = s_eff.groupby(gid).transform("max")
        survivors &= s_eff.eq(gmax)

    thr = float(delta_z_threshold or 0.0)
    if survivors.to_numpy(dtype=bool, na_value=False).any() and thr > 0.0:
        zvals = _to_numeric(out[z_col]).astype("float64")
        survivors = _collapse_within_dz(survivors, gid, zvals, crd_s, thr)

    n_surv_final = survivors.groupby(gid).transform("sum")
    one_winner = survivors & n_surv_final.eq(1)
    multi_winner = survivors & n_surv_final.ge(2)

    tr[one_winner.to_numpy(dtype=bool, na_value=False)] = 1
    tr[multi_winner.to_numpy(dtype=bool, na_value=False)] = 2

    out[tie_col] = pd.Series(tr, index=out.index).astype("Int8")

    if zf_series is not None:
        tr_num = pd.to_numeric(out[tie_col], errors="coerce")
        eq3_np = tr_num.eq(3.0).to_numpy(dtype=bool, na_value=False)
        is_star_np = is_star.to_numpy(dtype=bool, na_value=False)
        is_single_np = is_singleton.to_numpy(dtype=bool, na_value=False)

        invalid_3_np = eq3_np & ~is_star_np
        if invalid_3_np.any():
            invalid_3 = pd.Series(invalid_3_np, index=out.index)
            single = pd.Series(is_single_np, index=out.index)
            out.loc[invalid_3 & single, tie_col] = np.int8(1)
            out.loc[invalid_3 & ~single, tie_col] = np.int8(0)

    z_num = _to_numeric(out[z_col]).astype("float64")
    f_num = zf_num.fillna(-np.inf)
    t_num = (
        _score_instrument_type(
            out["instrument_type_homogenized"], instrument_type_priority
        ).astype("float64")
        if (
            "instrument_type_homogenized" in priority_set
            and instrument_type_priority is not None
        )
        else None
    )
    thr_local = float(delta_z_threshold or 0.0)

    is_pair = group_sizes.eq(2)
    pair_one = out[tie_col].eq(1) & is_pair
    both_one = pair_one.groupby(gid).transform("sum").eq(2)
    bad_gids = pd.Index(gid[pair_one & both_one]).unique()

    if len(bad_gids) > 0:
        for gval in bad_gids:
            mask = gid.eq(gval) & pair_one
            pos = np.flatnonzero(mask.to_numpy(dtype=bool, na_value=False))
            if pos.size != 2:
                continue
            p1, p2 = int(pos[0]), int(pos[1])

            f1, f2 = float(f_num.iloc[p1]), float(f_num.iloc[p2])
            if f1 > f2:
                out.iloc[[p1, p2], out.columns.get_loc(tie_col)] = np.array(
                    [1, 0], dtype=np.int8
                )
                continue
            if f2 > f1:
                out.iloc[[p1, p2], out.columns.get_loc(tie_col)] = np.array(
                    [0, 1], dtype=np.int8
                )
                continue

            if t_num is not None:
                t1, t2 = float(t_num.iloc[p1]), float(t_num.iloc[p2])
                if t1 > t2:
                    out.iloc[[p1, p2], out.columns.get_loc(tie_col)] = np.array(
                        [1, 0], dtype=np.int8
                    )
                    continue
                if t2 > t1:
                    out.iloc[[p1, p2], out.columns.get_loc(tie_col)] = np.array(
                        [0, 1], dtype=np.int8
                    )
                    continue

            z1, z2 = z_num.iloc[p1], z_num.iloc[p2]
            if thr_local <= 0.0 or pd.isna(z1) or pd.isna(z2):
                out.iloc[[p1, p2], out.columns.get_loc(tie_col)] = np.array(
                    [2, 2], dtype=np.int8
                )
                continue

            dz = abs(float(z1) - float(z2))
            if dz > thr_local:
                out.iloc[[p1, p2], out.columns.get_loc(tie_col)] = np.array(
                    [2, 2], dtype=np.int8
                )
            else:
                c1, c2 = str(crd_s.iloc[p1]), str(crd_s.iloc[p2])
                if c1 <= c2:
                    out.iloc[[p1, p2], out.columns.get_loc(tie_col)] = np.array(
                        [1, 0], dtype=np.int8
                    )
                else:
                    out.iloc[[p1, p2], out.columns.get_loc(tie_col)] = np.array(
                        [0, 1], dtype=np.int8
                    )

    pair_zero = out[tie_col].eq(0) & is_pair
    both_zero = pair_zero.groupby(gid).transform("sum").eq(2)
    bad_gids0 = pd.Index(gid[pair_zero & both_zero]).unique()

    if len(bad_gids0) > 0:
        for gval in bad_gids0:
            mask = gid.eq(gval) & pair_zero
            pos = np.flatnonzero(mask.to_numpy(dtype=bool, na_value=False))
            if pos.size != 2:
                continue
            p1, p2 = int(pos[0]), int(pos[1])

            f1, f2 = float(f_num.iloc[p1]), float(f_num.iloc[p2])
            if f1 > f2:
                out.iloc[p1, out.columns.get_loc(tie_col)] = np.int8(1)
                out.iloc[p2, out.columns.get_loc(tie_col)] = np.int8(0)
                continue
            if f2 > f1:
                out.iloc[p1, out.columns.get_loc(tie_col)] = np.int8(0)
                out.iloc[p2, out.columns.get_loc(tie_col)] = np.int8(1)
                continue

            if t_num is not None:
                t1, t2 = float(t_num.iloc[p1]), float(t_num.iloc[p2])
                if t1 > t2:
                    out.iloc[p1, out.columns.get_loc(tie_col)] = np.int8(1)
                    out.iloc[p2, out.columns.get_loc(tie_col)] = np.int8(0)
                    continue
                if t2 > t1:
                    out.iloc[p1, out.columns.get_loc(tie_col)] = np.int8(0)
                    out.iloc[p2, out.columns.get_loc(tie_col)] = np.int8(1)
                    continue

            z1, z2 = z_num.iloc[p1], z_num.iloc[p2]
            if (thr_local > 0.0) and not (pd.isna(z1) or pd.isna(z2)):
                dz = abs(float(z1) - float(z2))
                if dz <= thr_local:
                    c1, c2 = str(crd_s.iloc[p1]), str(crd_s.iloc[p2])
                    if c1 <= c2:
                        out.iloc[p1, out.columns.get_loc(tie_col)] = np.int8(1)
                        out.iloc[p2, out.columns.get_loc(tie_col)] = np.int8(0)
                    else:
                        out.iloc[p1, out.columns.get_loc(tie_col)] = np.int8(0)
                        out.iloc[p2, out.columns.get_loc(tie_col)] = np.int8(1)
                else:
                    out.iloc[[p1, p2], out.columns.get_loc(tie_col)] = np.array(
                        [2, 2], dtype=np.int8
                    )
            else:
                out.iloc[[p1, p2], out.columns.get_loc(tie_col)] = np.array(
                    [2, 2], dtype=np.int8
                )

    out = _apply_guard_restore_local(
        out,
        crd_col=crd_col,
        compared_col=compared_col,
        zf_series=zf_series,
        tie_col=tie_col,
        tie_col_orig=tie_col_orig,
    )

    drop_cols = []
    if tie_col_orig in out.columns:
        drop_cols.append(tie_col_orig)

    if group_col:
        out[group_col] = _assign_canonical_group_ids(out, "__group__", crd_col)
    else:
        drop_cols.append("__group__")

    out.drop(columns=drop_cols, inplace=True, errors="ignore")

    if zf_series is not None:
        out.loc[zf_series.eq(6).fillna(False), tie_col] = np.int8(3)

    if group_col:
        _validate_local_tie_invariants(
            out,
            group_col=group_col,
            tie_col=tie_col,
        )

    return out


# -----------------------
# LSDB per-partition dedup (with or without margin)
# -----------------------


def _ensure_string_pyarrow(s: pd.Series) -> pd.Series:
    """Cast to Arrow-backed string if available, else pandas string."""
    try:
        return s.astype("string[pyarrow]", copy=False)
    except Exception:
        return s.astype("string", copy=False)


def _nullable_int8(s: pd.Series) -> pd.Series:
    """Cast to pandas nullable Int8."""
    return s.astype("Int8", copy=False)


def _to_pandas(df):
    """Return a pandas.DataFrame for NestedFrame/unknown inputs."""
    if df is None:
        return pd.DataFrame()
    if hasattr(df, "to_pandas"):
        try:
            return df.to_pandas()
        except Exception:
            pass
    return pd.DataFrame(df)


def _shrink_to_needed(df, needed, crd_col, compared_col, z_col):
    """Project to required columns and normalize essential dtypes."""
    if df is None or len(df) == 0:
        cols = list(needed)
        out = pd.DataFrame({c: pd.Series(dtype="float64") for c in cols})
        for c in (crd_col, compared_col):
            out[c] = out[c].astype("string")
        return out[list(needed)]

    keep = [c for c in df.columns if c in needed]
    out = df[keep].copy()

    if crd_col not in out:
        out[crd_col] = pd.Series(dtype="string")
    if compared_col not in out:
        out[compared_col] = pd.Series(dtype="string")
    if z_col not in out:
        out[z_col] = pd.Series(dtype="float64")

    out[crd_col] = _ensure_string_pyarrow(out[crd_col])
    out[compared_col] = _ensure_string_pyarrow(out[compared_col])
    out[z_col] = pd.to_numeric(out[z_col], errors="coerce")

    for c in needed:
        if c not in out:
            out[c] = pd.Series(dtype="float64")

    return out[[c for c in needed]]


def _dedup_local_with_margin(
    part_main,
    part_margin,
    pixel,  # diagnostics only (unused for now)
    *,
    tiebreaking_priority: Sequence[str],
    instrument_type_priority: Mapping[str, int] | None,
    delta_z_threshold: float = 0.0,
    crd_col: str = "CRD_ID",
    compared_col: str = "compared_to",
    z_col: str = "z",
    tie_col: str = "tie_result",
    edge_log: bool = False,
    group_col: str | None = None,
    margin_threshold_arcsec: float = 5.0,
    margin_warning_fraction: float = 0.8,
) -> pd.DataFrame:
    """Run dedup on (main + margin) and return labels for main rows only.

    Args:
        part_main: Main partition (NestedFrame/pandas-like).
        part_margin: Margin partition aligned to main.
        pixel: Partition diagnostics (unused).
        tiebreaking_priority: Ordered columns for tie-breaking.
        instrument_type_priority: Map for instrument type scoring (optional).
        delta_z_threshold: Dz threshold for final disambiguation.
        crd_col: Name of the ID column.
        compared_col: Name of the neighbors column.
        z_col: Name of the redshift column.
        tie_col: Name of the output tie-result column.
        edge_log: Enable edge diagnostics during graph build.
        group_col: If set, also emit component/group id in this column.

    Returns:
        pd.DataFrame: Labels for main rows only.
    """
    # Normalize inputs to pandas.
    pm = _to_pandas(part_main)
    mg = _to_pandas(part_margin)

    # Project to required columns.
    needed = {crd_col, compared_col, z_col, tie_col, "ra", "dec"} | set(
        tiebreaking_priority or []
    )
    if instrument_type_priority is not None:
        needed.add("instrument_type_homogenized")
    pm = _shrink_to_needed(pm, needed, crd_col, compared_col, z_col)
    mg = _shrink_to_needed(mg, needed, crd_col, compared_col, z_col)

    # Early exit on empty view.
    if pm.empty and mg.empty:
        cols = {
            crd_col: pd.Series(dtype="string[pyarrow]"),
            tie_col: pd.Series(dtype="Int8"),
        }
        if group_col:
            cols[group_col] = pd.Series(dtype="Int64")
        return pd.DataFrame(cols)

    # Tag partition (cheap fingerprint for logs).
    try:
        i_min, i_max = pm.index.min(), pm.index.max()
        partition_tag = f"idx=[{i_min}..{i_max}]"
    except Exception:
        partition_tag = "idx=[unknown]"

    # Build view and run solver (guard-restore happens inside).
    pm["_src"] = "main"
    mg["_src"] = "margin"

    # Build the combined view while avoiding empty entries in concat.
    # This keeps pandas happy and future-proof w.r.t. all-NA/empty inputs.
    frames: list[pd.DataFrame] = []
    if not pm.empty:
        frames.append(pm)
    if not mg.empty:
        frames.append(mg)

    # This should not happen because we early-exit when both are empty,
    # but keep a defensive guard here for robustness.
    if not frames:
        cols = {
            crd_col: pd.Series(dtype="string[pyarrow]"),
            tie_col: pd.Series(dtype="Int8"),
        }
        if group_col:
            cols[group_col] = pd.Series(dtype="Int64")
        return pd.DataFrame(cols)

    view = pd.concat(frames, ignore_index=True)
    solved = deduplicate_pandas(
        view,
        tiebreaking_priority=tiebreaking_priority,
        instrument_type_priority=instrument_type_priority,
        delta_z_threshold=float(delta_z_threshold),
        crd_col=crd_col,
        compared_col=compared_col,
        z_col=z_col,
        tie_col=tie_col,
        edge_log=edge_log,
        partition_tag=partition_tag,
        logger=_phase_logger(),
        group_col=group_col,
    )

    if group_col and group_col in solved.columns:
        # Every non-star edge whose endpoints are present in this local view must
        # resolve to one canonical component.  This catches graph/label drift at
        # the partition boundary before labels are merged globally.
        zf = pd.to_numeric(solved.get("z_flag_homogenized"), errors="coerce")
        edge_nodes, edge_uv, _ = _build_edges_fast(
            solved,
            crd_col=crd_col,
            compared_col=compared_col,
            zf_series=zf,
            edge_log=False,
        )
        if edge_uv.size:
            group_by_id = solved.drop_duplicates(crd_col).set_index(crd_col)[group_col]
            left = group_by_id.reindex(edge_nodes.take(edge_uv[:, 0])).to_numpy()
            right = group_by_id.reindex(edge_nodes.take(edge_uv[:, 1])).to_numpy()
            if np.any(left != right):
                raise RuntimeError(
                    f"{partition_tag}: non-star edge endpoints received different group_id values"
                )

        src_counts = solved.groupby(group_col)["_src"].nunique()
        boundary_groups = src_counts[src_counts > 1].index
        if len(boundary_groups):
            lg = _phase_logger()
            lg.info(
                "%s Boundary components touching main+margin: count=%d sample_group_ids=%s",
                partition_tag,
                len(boundary_groups),
                list(boundary_groups[:5]),
            )

            # A component approaching the full margin width may be truncated in
            # another pixel.  Use the diagonal of its local RA/Dec bounding box
            # as a cheap conservative diagnostic (RA adjusted by cos(dec)).
            warn_at = float(margin_threshold_arcsec) * float(margin_warning_fraction)
            near_limit = []
            boundary_rows = solved[solved[group_col].isin(boundary_groups)]
            for gid_value, component in boundary_rows.groupby(group_col):
                ra = pd.to_numeric(component["ra"], errors="coerce")
                dec = pd.to_numeric(component["dec"], errors="coerce")
                valid = ra.notna() & dec.notna()
                if valid.sum() < 2:
                    continue
                dec_mid = math.radians(float(dec[valid].mean()))
                dra = float(ra[valid].max() - ra[valid].min())
                dra = min(dra, 360.0 - dra) * math.cos(dec_mid)
                ddec = float(dec[valid].max() - dec[valid].min())
                extent_arcsec = math.hypot(dra, ddec) * 3600.0
                if extent_arcsec >= warn_at:
                    near_limit.append((int(gid_value), extent_arcsec))
            if near_limit:
                lg.warning(
                    "%s Components approach/exceed margin support %.3f arcsec: count=%d sample=%s",
                    partition_tag,
                    float(margin_threshold_arcsec),
                    len(near_limit),
                    near_limit[:5],
                )

    # Keep only main rows and required columns.
    cols = [crd_col, tie_col]
    if group_col and (group_col in solved.columns):
        cols.append(group_col)

    out = solved.loc[solved["_src"] == "main", cols].copy()

    # Stable dtypes.
    out[crd_col] = _ensure_string_pyarrow(out[crd_col])
    out[tie_col] = _nullable_int8(out[tie_col])
    if group_col and (group_col in out.columns):
        out[group_col] = out[group_col].astype("Int64")

    return out


def _dedup_alignfunc_with_margin(
    part_main,
    part_margin,
    pixel_main,
    pixel_margin,
    info_main,
    info_margin,
    *,
    tiebreaking_priority: Sequence[str],
    instrument_type_priority: Mapping[str, int] | None,
    delta_z_threshold: float = 0.0,
    crd_col: str = "CRD_ID",
    compared_col: str = "compared_to",
    z_col: str = "z",
    tie_col: str = "tie_result",
    edge_log: bool = False,
    group_col: str | None = None,
    margin_threshold_arcsec: float = 5.0,
    margin_warning_fraction: float = 0.8,
) -> pd.DataFrame:
    """Adapter for LSDB/HATS `align_and_apply`.

    Args:
        part_main: Main partition (NestedFrame/pandas-like).
        part_margin: Margin partition aligned to the same HATS pixel.
        pixel_main: Healpix pixel for the main partition (diagnostics only).
        pixel_margin: Healpix pixel for the margin partition (unused).
        info_main: Catalog properties for the main catalog (unused).
        info_margin: Catalog properties for the margin catalog (unused).

    Returns:
        pd.DataFrame: Labels for main rows only.
    """
    return _dedup_local_with_margin(
        part_main,
        part_margin,
        pixel_main,  # diagnostics only
        tiebreaking_priority=tiebreaking_priority,
        instrument_type_priority=instrument_type_priority,
        delta_z_threshold=delta_z_threshold,
        crd_col=crd_col,
        compared_col=compared_col,
        z_col=z_col,
        tie_col=tie_col,
        edge_log=edge_log,
        group_col=group_col,
        margin_threshold_arcsec=margin_threshold_arcsec,
        margin_warning_fraction=margin_warning_fraction,
    )


def _dedup_local_no_margin(
    part_main,
    *,
    tiebreaking_priority: Sequence[str],
    instrument_type_priority: Mapping[str, int] | None,
    delta_z_threshold: float = 0.0,
    crd_col: str = "CRD_ID",
    compared_col: str = "compared_to",
    z_col: str = "z",
    tie_col: str = "tie_result",
    edge_log: bool = False,
    group_col: str | None = None,
) -> pd.DataFrame:
    """Run dedup using only the main partition.

    Args:
        part_main: Main partition (NestedFrame/pandas-like).
        tiebreaking_priority: Ordered columns for tie-breaking.
        instrument_type_priority: Map for instrument type scoring (optional).
        delta_z_threshold: Dz threshold for final disambiguation.
        crd_col: Name of the ID column.
        compared_col: Name of the neighbors column.
        z_col: Name of the redshift column.
        tie_col: Name of the output tie-result column.
        edge_log: Enable edge diagnostics during graph build.
        group_col: If set, also emit component/group id in this column.

    Returns:
        pd.DataFrame: Labels for main rows only.
    """
    # Normalize inputs to pandas.
    pm = _to_pandas(part_main)

    # Project to required columns.
    needed = {crd_col, compared_col, z_col, tie_col} | set(tiebreaking_priority or [])
    if instrument_type_priority is not None:
        needed.add("instrument_type_homogenized")
    pm = _shrink_to_needed(pm, needed, crd_col, compared_col, z_col)

    # Early exit on empty view.
    if pm.empty:
        cols = {
            crd_col: pd.Series(dtype="string[pyarrow]"),
            tie_col: pd.Series(dtype="Int8"),
        }
        if group_col:
            cols[group_col] = pd.Series(dtype="Int64")
        return pd.DataFrame(cols)

    # Tag partition (cheap fingerprint for logs).
    try:
        i_min, i_max = pm.index.min(), pm.index.max()
        partition_tag = f"idx=[{i_min}..{i_max}]"
    except Exception:
        partition_tag = "idx=[unknown]"

    # Run solver (guard-restore happens inside).
    solved = deduplicate_pandas(
        pm,
        tiebreaking_priority=tiebreaking_priority,
        instrument_type_priority=instrument_type_priority,
        delta_z_threshold=float(delta_z_threshold),
        crd_col=crd_col,
        compared_col=compared_col,
        z_col=z_col,
        tie_col=tie_col,
        edge_log=edge_log,
        partition_tag=partition_tag,
        logger=_phase_logger(),
        group_col=group_col,
    )

    # Select output columns.
    cols = [crd_col, tie_col]
    if group_col and (group_col in solved.columns):
        cols.append(group_col)

    out = solved[cols].copy()

    # Stable dtypes.
    out[crd_col] = _ensure_string_pyarrow(out[crd_col])
    out[tie_col] = _nullable_int8(out[tie_col])
    if group_col and (group_col in out.columns):
        out[group_col] = out[group_col].astype("Int64")

    return out


def _columns_set(ddf) -> set:
    """Return the set of column names from a Dask DataFrame's _meta (cheap)."""
    return set(map(str, getattr(getattr(ddf, "_meta", None), "columns", [])))


def _assert_required(ddf, required: set[str], where: str):
    """Raise KeyError if required columns are missing from ddf."""
    have = _columns_set(ddf)
    missing = sorted([c for c in required if c not in have])
    if missing:
        raise KeyError(f"Missing required columns in {where}: {missing}")


def _assert_priorities(ddf, priorities: list[str], where: str):
    """Raise KeyError if any tiebreaking_priority column is missing from ddf."""
    have = _columns_set(ddf)
    missing = sorted([c for c in priorities if c not in have])
    if missing:
        raise KeyError(f"Missing priority columns in {where}: {missing}")


def run_dedup_with_lsdb_map_partitions(
    cat,
    *,
    tiebreaking_priority: Sequence[str],
    instrument_type_priority: Mapping[str, int] | None,
    delta_z_threshold: float = 0.0,
    crd_col: str = "CRD_ID",
    compared_col: str = "compared_to",
    z_col: str = "z",
    tie_col: str = "tie_result",
    edge_log: bool = False,
    group_col: str | None = None,  # new
    margin_threshold_arcsec: float = 5.0,
    margin_warning_fraction: float = 0.8,
) -> dd.DataFrame:
    """Compute dedup labels per partition via LSDB; align divisions if margin exists.

    Args:
        cat: LSDB catalog.
        tiebreaking_priority: Ordered columns for tie-breaking.
        instrument_type_priority: Map for instrument type scoring (optional).
        delta_z_threshold: Dz threshold for final disambiguation.
        crd_col: Name of the ID column.
        compared_col: Name of the neighbors column.
        z_col: Name of the redshift column.
        tie_col: Name of the output tie-result column.
        edge_log: Enable edge diagnostics during graph build.
        group_col: If set, also emit component/group id in this column.

    Returns:
        dd.DataFrame: Dask DataFrame with tie labels (and optional group ids).
    """
    with log_phase(
        "deduplication", "run_dedup_with_lsdb_map_partitions", _base_logger()
    ) as log:
        # --- Early validations ---
        if not hasattr(cat, "_ddf"):
            raise AttributeError("Catalog does not expose _ddf.")
        if (
            not isinstance(tiebreaking_priority, (list, tuple))
            or len(tiebreaking_priority) == 0
        ):
            raise TypeError("tiebreaking_priority must be a non-empty sequence.")
        if float(delta_z_threshold) < 0.0:
            raise ValueError("delta_z_threshold must be non-negative.")

        main_ddf = cat._ddf
        has_margin = bool(getattr(cat, "margin", None) and hasattr(cat.margin, "_ddf"))
        log.info(
            "Inputs: has_margin=%s, npartitions(main)=%s",
            has_margin,
            main_ddf.npartitions,
        )

        # Strict schema checks: required base + all tiebreaking priority columns
        required_base = {crd_col, compared_col, z_col}
        _assert_required(main_ddf, required_base, "main")
        _assert_priorities(main_ddf, list(tiebreaking_priority), "main")

        if has_margin:
            margin_ddf = cat.margin._ddf
            _assert_required(margin_ddf, required_base, "margin")
            _assert_priorities(margin_ddf, list(tiebreaking_priority), "margin")

        # Enforce instrument_type mapping only if requested in priorities
        if ("instrument_type_homogenized" in set(tiebreaking_priority)) and (
            instrument_type_priority is None
        ):
            raise ValueError(
                "instrument_type_priority is required when "
                "'instrument_type_homogenized' is used in tiebreaking_priority."
            )

        # Output meta for map_partitions
        meta_dict = {
            crd_col: pd.Series(dtype="string[pyarrow]"),
            tie_col: pd.Series(dtype="Int8"),
        }
        if group_col:
            meta_dict[group_col] = pd.Series(dtype="Int64")
        meta = pd.DataFrame(meta_dict)

        if not has_margin:
            log.info("No margin attached; running local dedup without margin.")
            labels_dd = main_ddf.map_partitions(
                _dedup_local_no_margin,
                meta=meta,
                tiebreaking_priority=tiebreaking_priority,
                instrument_type_priority=instrument_type_priority,
                delta_z_threshold=float(delta_z_threshold),
                crd_col=crd_col,
                compared_col=compared_col,
                z_col=z_col,
                tie_col=tie_col,
                edge_log=edge_log,
                group_col=group_col,
            )
        else:
            # ------------------------------------------------------------------
            # NEW: margin-aware path using LSDB/HATS pixel-tree alignment.
            #
            # Instead of forcing Dask divisions of `main_ddf` and `margin_ddf`
            # to match, we rely on the HATS pixel-tree alignment used by LSDB
            # for concatenation. This guarantees that main and margin partitions
            # are aligned on the same HATS cells (including pixels that appear
            # only in the margin).
            # ------------------------------------------------------------------
            log.info(
                "Margin attached; running dedup using LSDB/HATS pixel alignment "
                "(concat_align_catalogs + align_and_apply)."
            )

            # Build a concatenation-oriented pixel alignment between `cat`
            # and its margin. We disable MOC filtering so that pixels that
            # appear only in the margin are still included.
            with log_phase(
                "deduplication", "pixel_alignment_main_margin", _base_logger()
            ):
                # Use the same catalog on both sides so concat_align_catalogs can
                # expand main + margin internally. Passing `cat.margin` directly
                # would fail because MarginCatalog has no `.margin` attribute.
                alignment = concat_align_catalogs(
                    cat,
                    cat,
                    filter_by_mocs=False,
                    # alignment_type default is OUTER, which is what we want
                )
                aligned_pixels = get_aligned_pixels_from_alignment(alignment)
                log.info(
                    "Pixel alignment built for main+margin: n_aligned_pixels=%d",
                    len(aligned_pixels),
                )

            # Each entry in `catalog_mappings` is:
            #   (HealpixDataset, list[HealpixPixel])
            # Here we align the main catalog and its margin on the same pixel set.
            catalog_mappings = [
                (cat, aligned_pixels),
                (cat.margin, aligned_pixels),
            ]

            # Align partitions on the HATS pixel grid and run dedup on
            # (main + margin) for each pixel using the adapter defined above.
            with log_phase(
                "deduplication", "align_and_apply_with_margin", _base_logger()
            ):
                delayed_parts = align_and_apply(
                    catalog_mappings,
                    func=_dedup_alignfunc_with_margin,
                    tiebreaking_priority=tiebreaking_priority,
                    instrument_type_priority=instrument_type_priority,
                    delta_z_threshold=float(delta_z_threshold),
                    crd_col=crd_col,
                    compared_col=compared_col,
                    z_col=z_col,
                    tie_col=tie_col,
                    edge_log=edge_log,
                    group_col=group_col,
                    margin_threshold_arcsec=float(margin_threshold_arcsec),
                    margin_warning_fraction=float(margin_warning_fraction),
                )

            # Build a Dask DataFrame from the delayed per-pixel label frames.
            labels_dd = dd.from_delayed(delayed_parts, meta=meta)

        # Stable dtypes for downstream merges.
        assign_map = {
            crd_col: labels_dd[crd_col].astype("string[pyarrow]"),
            tie_col: labels_dd[tie_col].astype("Int8"),
        }
        if group_col:
            assign_map[group_col] = labels_dd[group_col].astype("Int64")
        labels_dd = labels_dd.assign(**assign_map)

        log.info("Labels dtypes stabilized.")
        return labels_dd
