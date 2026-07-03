from __future__ import annotations

"""Self-crossmatch and `compared_to` updater for CRC.

Runs a self spatial crossmatch on a single catalog, updates its `compared_to`
column (adds neighbor CRD_IDs, removes self-links), and writes a new on-disk
collection `<artifact>_auto`.

Public API:
    - crossmatch_auto
"""

# -----------------------
# Standard library
# -----------------------
import logging
import os
import time
from typing import TYPE_CHECKING, Dict, Iterable, List, Set

if TYPE_CHECKING:
    import lsdb

# -----------------------
# Third-party
# -----------------------
import numpy as np
import pandas as pd

# -----------------------
# Project
# -----------------------
from specz import DTYPE_STR  # Arrow-backed string dtype
from crossmatch_diagnostics import (
    log_component_size_diagnostics,
    log_neighbor_count_diagnostics,
    log_pair_separation_diagnostics,
)
from utils import get_phase_logger

__all__ = ["crossmatch_auto"]

LOGGER_NAME = "crc.crossmatch_auto"  # child of the central pipeline logger


# -----------------------
# Logging helper
# -----------------------
def _get_logger() -> logging.LoggerAdapter:
    """Return a phase-aware logger ('crc.crossmatch_auto' with phase='automatch').

    Returns:
        logging.LoggerAdapter: Logger with phase context.
    """
    base = logging.getLogger(LOGGER_NAME)
    base.setLevel(logging.NOTSET)
    base.propagate = True
    return get_phase_logger("automatch", base)


# -----------------------
# Internal helpers
# -----------------------
def _is_collection_root(path: str) -> bool:
    """Return True when `path` looks like a complete HATS collection root."""
    return (
        bool(path)
        and os.path.isdir(path)
        and os.path.exists(os.path.join(path, "collection.properties"))
    )


def _adjacency_from_pairs(
    left_ids: pd.Series, right_ids: pd.Series
) -> Dict[str, Set[str]]:
    """Build an undirected adjacency from left-right crossmatch pairs.

    Args:
        left_ids: Series with left CRD_IDs.
        right_ids: Series with right CRD_IDs.

    Returns:
        Dict[str, Set[str]]: Mapping of CRD_ID to neighbor ids.
    """
    adj: Dict[str, Set[str]] = {}
    L = left_ids.astype(str).to_numpy(dtype=object, copy=False)
    R = right_ids.astype(str).to_numpy(dtype=object, copy=False)
    get = adj.get
    for a, b in zip(L, R):
        if a == b:
            continue
        s = get(a)
        if s is None:
            adj[a] = {b}
        else:
            s.add(b)
        s = get(b)
        if s is None:
            adj[b] = {a}
        else:
            s.add(a)
    return adj


def _log_neighbor_saturation(
    pairs: pd.DataFrame,
    *,
    limit: int,
    logger: logging.LoggerAdapter,
    total_by_source: dict | None,
    warn_fraction: float,
    fail_fraction: float | None,
) -> None:
    """Apply logging/failure policy to self-crossmatch saturation."""
    if pairs.empty or limit <= 0:
        return
    group_cols = ["CRD_IDleft"]
    if "sourceleft" in pairs:
        group_cols.insert(0, "sourceleft")
    counts = (
        pairs[group_cols + ["CRD_IDright"]]
        .drop_duplicates()
        .groupby(group_cols, dropna=False)
        .size()
    )
    if "sourceleft" in pairs:
        for source, source_counts in counts.groupby(level=0, dropna=False):
            log_neighbor_count_diagnostics(
                source_counts,
                limit=limit,
                logger=logger,
                context=f"Self-crossmatch source={source}",
            )
            total = int(total_by_source.get(str(source), source_counts.size))
            saturated = int(source_counts.ge(limit).sum())
            fraction = saturated / total if total else 0.0
            log_method = logger.warning if fraction >= warn_fraction else logger.info
            log_method(
                "Self-crossmatch neighbor saturation: source=%s at_limit=%d total_objects=%d fraction=%.6f limit=%d",
                source,
                saturated,
                total,
                fraction,
                limit,
            )
            if fail_fraction is not None and fraction >= fail_fraction:
                raise RuntimeError(
                    f"Self-crossmatch neighbor saturation for source={source} is "
                    f"{fraction:.6f}, reaching failure threshold {fail_fraction:.6f}; "
                    "increase crossmatch_n_neighbors"
                )
    else:
        log_neighbor_count_diagnostics(
            counts,
            limit=limit,
            logger=logger,
            context="Self-crossmatch",
        )
        total = int(total_by_source.get("<all>", counts.size))
        saturated = int(counts.ge(limit).sum())
        fraction = saturated / total if total else 0.0
        log_method = logger.warning if fraction >= warn_fraction else logger.info
        log_method(
            "Self-crossmatch neighbor saturation: at_limit=%d total_objects=%d fraction=%.6f limit=%d",
            saturated,
            total,
            fraction,
            limit,
        )
        if fail_fraction is not None and fraction >= fail_fraction:
            raise RuntimeError(
                f"Self-crossmatch neighbor saturation is {fraction:.6f}, reaching "
                f"failure threshold {fail_fraction:.6f}; increase crossmatch_n_neighbors"
            )


def _catalog_source_totals(catalog) -> dict:
    """Count all catalog objects by source for saturation denominators."""
    if "source" in catalog._ddf.columns:
        counts = catalog._ddf.groupby("source").size().compute()
        return {str(source): int(count) for source, count in counts.items()}
    return {"<all>": int(catalog._ddf.map_partitions(len).sum().compute())}


def _merge_compared_to_partition(
    part: pd.DataFrame,
    pairs_adj: Dict[str, Iterable[str]],
) -> pd.DataFrame:
    """Update one partition by unioning `compared_to` with new adjacency.

    Args:
        part: Partition dataframe.
        pairs_adj: Mapping CRD_ID -> iterable of neighbor ids.

    Returns:
        pd.DataFrame: Partition with updated `compared_to` (NA if empty).
    """
    p = part.copy()

    if "compared_to" not in p.columns:
        p["compared_to"] = pd.Series(
            pd.array([pd.NA] * len(p), dtype=DTYPE_STR), index=p.index
        )

    crd_list: List[str] = p["CRD_ID"].astype(str).tolist()

    def _norm_token(x) -> str | None:
        if pd.isna(x):
            return None
        if isinstance(x, (bool, np.bool_)):
            return None
        s = str(x).strip()
        if not s or s == "<NA>":
            return None
        return s

    def _to_str_set(it: Iterable) -> Set[str]:
        out: Set[str] = set()
        if it is None:
            return out
        for x in it:
            s = _norm_token(x)
            if s is not None:
                out.add(s)
        return out

    def _parse_existing(val) -> Set[str]:
        if isinstance(val, str):
            return _to_str_set(t.strip() for t in val.split(","))
        if isinstance(val, (list, set, tuple)):
            return _to_str_set(val)
        if pd.isna(val):
            return set()
        return _to_str_set([val])

    # New neighbor sets (Python-side).
    new_sets: List[Set[str]] = [_to_str_set(pairs_adj.get(k, ())) for k in crd_list]

    # Old compared_to as sets.
    old_sets: List[Set[str]] = [_parse_existing(v) for v in p["compared_to"].tolist()]

    # Union; drop self; stringify or keep NA.
    merged_vals: List[object] = []
    for k, old_set, new_set in zip(crd_list, old_sets, new_sets):
        nxt = set().union(old_set, new_set)
        nxt.discard(k)
        merged_vals.append(", ".join(sorted(nxt)) if nxt else pd.NA)

    p["compared_to"] = pd.Series(pd.array(merged_vals, dtype=DTYPE_STR), index=p.index)
    return p


def _self_xmatch_pairs(
    catalog: "lsdb.catalog.Catalog",
    radius_arcsec: float,
    n_neighbors: int,
    logger: logging.LoggerAdapter,
    total_by_source: dict,
    warn_fraction: float,
    fail_fraction: float | None,
) -> Dict[str, Set[str]]:
    """Run a self-crossmatch and return an adjacency (CRD_ID -> neighbor ids).

    Args:
        catalog: LSDB catalog to crossmatch with itself.
        radius_arcsec: Matching radius in arcseconds.
        n_neighbors: Maximum neighbors per source.
        logger: Logger.

    Returns:
        Dict[str, Set[str]]: Mapping of CRD_ID to neighbor ids.
    """
    logger.info(
        'Running self-crossmatch: radius=%.3f" n_neighbors=%d',
        radius_arcsec,
        n_neighbors,
    )
    xmatched = catalog.crossmatch(
        catalog,
        radius_arcsec=radius_arcsec,
        n_neighbors=n_neighbors,
        suffixes=("left", "right"),
        suffix_method='all_columns',
    )
    pair_cols = ["CRD_IDleft", "CRD_IDright"]
    if "_dist_arcsec" in xmatched.columns:
        pair_cols.append("_dist_arcsec")
    if total_by_source is not None and "sourceleft" in xmatched.columns:
        pair_cols.append("sourceleft")
    pairs_df = xmatched[pair_cols].compute()
    if len(pairs_df) == 0:
        logger.info("Self-crossmatch: no pairs found; `compared_to` remains unchanged.")
        return {}
    log_pair_separation_diagnostics(
        pairs_df,
        left_col="CRD_IDleft",
        right_col="CRD_IDright",
        radius_arcsec=radius_arcsec,
        logger=logger,
        context="Self-crossmatch",
    )
    if total_by_source is not None:
        _log_neighbor_saturation(
            pairs_df,
            limit=n_neighbors,
            logger=logger,
            total_by_source=total_by_source,
            warn_fraction=warn_fraction,
            fail_fraction=fail_fraction,
        )
    pairs_df = pairs_df.astype({"CRD_IDleft": "string", "CRD_IDright": "string"})
    pairs_df = pairs_df[
        pairs_df["CRD_IDleft"] != pairs_df["CRD_IDright"]
    ].drop_duplicates()

    adj = _adjacency_from_pairs(pairs_df["CRD_IDleft"], pairs_df["CRD_IDright"])
    log_component_size_diagnostics(
        adj,
        logger=logger,
        context="Self-crossmatch",
    )
    total_links = sum(len(v) for v in adj.values())
    logger.info(
        "Self-crossmatch: %d unique pairs across %d nodes", total_links, len(adj)
    )
    return adj


def _update_compared_to(
    catalog: "lsdb.catalog.Catalog",
    pairs_adj: Dict[str, Set[str]],
) -> "lsdb.catalog.Catalog":
    """Return a new Catalog with updated `compared_to` in each partition.

    Args:
        catalog: LSDB catalog.
        pairs_adj: Mapping CRD_ID -> neighbor ids.

    Returns:
        lsdb.catalog.Catalog: Catalog with updated `compared_to`.
    """
    meta_pdf = _ensure_compared_to_meta(catalog._ddf._meta)
    return catalog.map_partitions(
        _merge_compared_to_partition, pairs_adj, meta=meta_pdf
    )


def _ensure_compared_to_meta(meta_df: pd.DataFrame) -> pd.DataFrame:
    """Return a meta dataframe with a typed `compared_to` column.

    Args:
        meta_df: Input meta dataframe.

    Returns:
        pd.DataFrame: Meta dataframe with `compared_to` set to string dtype.
    """
    meta = meta_df.copy()
    meta["compared_to"] = pd.Series(pd.array([], dtype=DTYPE_STR))
    return meta


# -----------------------
# Public API
# -----------------------
def crossmatch_auto(
    catalog: "lsdb.catalog.Catalog",
    collection_path: str,
    logs_dir: str,  # kept for API compatibility; logging is centralized
    translation_config: dict | None = None,
) -> str:
    """Run self-crossmatch, update `compared_to`, and persist as `<artifact>_auto`.

    Args:
        catalog: LSDB catalog already imported as a collection.
        collection_path: Existing collection path (e.g., ".../temp/003_xxx_hats").
        logs_dir: Unused (logging is centralized via the root pipeline logger).
        translation_config: Optional dict with `crossmatch_radius_arcsec` and
            `crossmatch_n_neighbors`.

    Returns:
        str: Path to the new collection `<artifact>_auto`.
    """
    logger = _get_logger()
    if not collection_path:
        raise ValueError("`collection_path` must be provided (collection-only mode).")

    # Parameters with defaults
    radius = float((translation_config or {}).get("crossmatch_radius_arcsec", 0.75))
    k = int((translation_config or {}).get("crossmatch_n_neighbors", 20))
    saturation_enabled = bool(
        (translation_config or {}).get("crossmatch_saturation_enabled", False)
    )
    warn_fraction = 0.01
    fail_raw = None
    if saturation_enabled:
        warn_fraction = float(
            (translation_config or {}).get(
                "crossmatch_saturation_warn_fraction", 0.01
            )
        )
        fail_raw = (translation_config or {}).get(
            "crossmatch_saturation_fail_fraction", 0.10
        )
    fail_fraction = None if fail_raw is None else float(fail_raw)
    if saturation_enabled and not 0.0 <= warn_fraction <= 1.0:
        raise ValueError("crossmatch_saturation_warn_fraction must be in [0, 1]")
    if saturation_enabled and fail_fraction is not None and not warn_fraction <= fail_fraction <= 1.0:
        raise ValueError(
            "crossmatch_saturation_fail_fraction must be null or in "
            "[crossmatch_saturation_warn_fraction, 1]"
        )
    # Derive parent dir and artifact names
    parent_dir, artifact = os.path.split(os.path.normpath(collection_path))
    if not parent_dir:
        parent_dir = "."
    artifact_auto = f"{artifact}_auto"
    collection_path_auto = os.path.join(parent_dir, artifact_auto)

    if _is_collection_root(collection_path_auto):
        logger.info(
            "Skip automatch: output already exists and is a valid collection: %s",
            collection_path_auto,
        )
        return collection_path_auto

    total_by_source = _catalog_source_totals(catalog) if saturation_enabled else {}

    # START (per-catalog)
    t0 = time.time()
    logger.info(
        'START automatch: artifact=%s radius=%.3f" n_neighbors=%d src=%s dst=%s',
        artifact,
        radius,
        k,
        collection_path,
        collection_path_auto,
    )

    # 1) Self-crossmatch -> adjacency (CRD_ID -> neighbor ids)
    pairs_adj = _self_xmatch_pairs(
        catalog,
        radius,
        k,
        logger,
        total_by_source if saturation_enabled else None,
        warn_fraction,
        fail_fraction,
    )

    # 2) Update `compared_to`
    updated = _update_compared_to(catalog, pairs_adj)
    total_links = sum(len(v) for v in pairs_adj.values())
    n_nodes = len(pairs_adj)

    # 3) Persist as a NEW collection in the parent directory
    logger.info(
        "Writing collection: base_dir=%s catalog_name=%s", parent_dir, artifact_auto
    )
    updated.write_catalog(
        collection_path_auto,
        as_collection=True,
        overwrite=True,
    )
    logger.info("Write complete: path=%s", collection_path_auto)

    # END (per-catalog)
    dt = time.time() - t0
    logger.info(
        "END automatch: artifact=%s links=%d nodes=%d output=%s (%.2fs)",
        artifact,
        total_links,
        n_nodes,
        collection_path_auto,
        dt,
    )

    return collection_path_auto
