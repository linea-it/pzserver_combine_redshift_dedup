"""Low-cost diagnostics over already-materialized crossmatch results."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from collections.abc import Set as AbstractSet

import dask.dataframe as dd
import numpy as np
import pandas as pd


def project_catalog_for_pair_crossmatch(catalog, *, include_source: bool):
    """Return a narrow LSDB catalog, preserving its projected margin.

    LSDB crossmatch outputs include every input column.  Projecting only after
    the crossmatch is too late for very large partitions because Distributed
    may need to serialize the wide ``NestedFrame`` dependency first.  Narrowing
    both the main catalog and its margin here prevents that wide result from
    being created at all.
    """
    required = ["CRD_ID", "ra", "dec"]
    if include_source and "source" in catalog.columns:
        required.append("source")

    missing = [column for column in required if column not in catalog.columns]
    if missing:
        raise KeyError(f"Crossmatch pair projection is missing columns: {missing}")

    projected = catalog[required]
    margin = getattr(catalog, "margin", None)
    if margin is not None:
        margin_columns = [column for column in required if column in margin.columns]
        projected.margin = margin[margin_columns]
    return projected


def _project_pairs_partition(part, columns: list[str]) -> pd.DataFrame:
    """Return only pair columns as a plain pandas frame.

    LSDB partitions are ``NestedFrame`` instances.  Returning one of those from
    a distributed ``compute`` can make Distributed pickle every crossmatch
    column, even after a lazy column selection.  Materializing the projection
    inside the worker keeps the transfer narrow and avoids NestedFrame
    serialization.
    """
    projected = part.loc[:, columns]
    return pd.DataFrame(
        {column: projected[column] for column in columns},
        index=projected.index,
    )


def compute_projected_pairs(ddf, columns: list[str]) -> pd.DataFrame:
    """Compute a worker-side pair projection and return a plain pandas frame."""
    meta = _project_pairs_partition(ddf._meta, columns).iloc[:0]
    projected = ddf.map_partitions(
        _project_pairs_partition,
        columns,
        meta=meta,
    )
    result = projected.compute()
    if type(result) is not pd.DataFrame:
        result = pd.DataFrame(result)
    return result


def stage_projected_pairs(ddf, columns: list[str], path: str):
    """Project pair columns inside workers and stage them as Parquet.

    Unlike :func:`compute_projected_pairs`, this never gathers all pair rows on
    the driver. The returned Dask dataframe starts from the staged dataset, so
    downstream shuffles also avoid retaining the original LSDB graph.
    """
    meta = _project_pairs_partition(ddf._meta, columns).iloc[:0]
    projected = ddf.map_partitions(
        _project_pairs_partition,
        columns,
        meta=meta,
    )
    projected.to_parquet(path, engine="pyarrow", write_index=False)
    return dd.read_parquet(path, engine="pyarrow")


def log_pair_separation_diagnostics(
    pairs: pd.DataFrame,
    *,
    left_col: str,
    right_col: str,
    radius_arcsec: float,
    logger: logging.LoggerAdapter,
    context: str,
    distance_col: str = "_dist_arcsec",
) -> None:
    """Log edge-distance quantiles using the LSDB distance already in memory."""
    if distance_col not in pairs.columns or pairs.empty:
        return
    nonself = pairs[left_col].astype(str).ne(pairs[right_col].astype(str))
    distances = pd.to_numeric(
        pairs.loc[nonself, distance_col], errors="coerce"
    ).to_numpy(dtype="float64", na_value=np.nan)
    distances = distances[np.isfinite(distances)]
    if distances.size == 0:
        return

    p50, p90, p95, p99 = np.quantile(distances, [0.50, 0.90, 0.95, 0.99])
    radius = float(radius_arcsec)
    logger.info(
        "%s separation diagnostics: pairs=%d p50=%.4farcsec "
        "p90=%.4farcsec p95=%.4farcsec p99=%.4farcsec max=%.4farcsec "
        "fraction_gt_50pct_radius=%.6f fraction_gt_80pct_radius=%.6f "
        "fraction_gt_95pct_radius=%.6f",
        context,
        int(distances.size),
        float(p50),
        float(p90),
        float(p95),
        float(p99),
        float(distances.max()),
        float(np.mean(distances > 0.50 * radius)),
        float(np.mean(distances > 0.80 * radius)),
        float(np.mean(distances > 0.95 * radius)),
    )


def log_neighbor_count_diagnostics(
    counts: pd.Series,
    *,
    limit: int,
    logger: logging.LoggerAdapter,
    context: str,
) -> None:
    """Log quantiles of returned matches per left-side object."""
    values = pd.to_numeric(counts, errors="coerce").dropna().to_numpy(dtype="float64")
    if values.size == 0:
        return
    p50, p90, p99 = np.quantile(values, [0.50, 0.90, 0.99])
    logger.info(
        "%s returned-match diagnostics: objects_with_matches=%d "
        "p50=%.1f p90=%.1f p99=%.1f max=%d fraction_ge_2=%.6f "
        "fraction_ge_5=%.6f fraction_at_limit=%.6f limit=%d",
        context,
        int(values.size),
        float(p50),
        float(p90),
        float(p99),
        int(values.max()),
        float(np.mean(values >= 2)),
        float(np.mean(values >= 5)),
        float(np.mean(values >= limit)),
        int(limit),
    )


def log_component_size_diagnostics(
    adjacency: Mapping[str, AbstractSet[str]],
    *,
    logger: logging.LoggerAdapter,
    context: str,
) -> None:
    """Log connected-component sizes with one linear traversal of adjacency."""
    if not adjacency:
        return
    visited: set[str] = set()
    sizes: list[int] = []
    densities: list[float] = []
    tree_like = 0
    for root in adjacency:
        if root in visited:
            continue
        visited.add(root)
        stack = [root]
        size = 0
        degree_sum = 0
        while stack:
            node = stack.pop()
            size += 1
            neighbors = adjacency.get(node, ())
            degree_sum += len(neighbors)
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        sizes.append(size)
        edges = degree_sum // 2
        if size >= 3:
            possible_edges = size * (size - 1) / 2
            densities.append(edges / possible_edges)
            tree_like += int(edges == size - 1)

    values = np.asarray(sizes, dtype="int64")
    p50, p90, p99 = np.quantile(values, [0.50, 0.90, 0.99])
    logger.info(
        "%s component-size diagnostics: components=%d nodes=%d "
        "p50=%.1f p90=%.1f p99=%.1f max=%d components_ge_3=%d "
        "components_ge_10=%d tree_like_ge_3=%d low_density_lt_0.5_ge_3=%d "
        "density_p50_ge_3=%.4f",
        context,
        int(values.size),
        int(values.sum()),
        float(p50),
        float(p90),
        float(p99),
        int(values.max()),
        int(np.sum(values >= 3)),
        int(np.sum(values >= 10)),
        tree_like,
        int(np.sum(np.asarray(densities) < 0.5)),
        float(np.median(densities)) if densities else 1.0,
    )
