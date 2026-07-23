"""Orchestrate the CRC pipeline (prepare -> auto-crossmatch -> crossmatch -> deduplicate -> export).

Public API:
    - main
"""

from __future__ import annotations

# -----------------------
# Standard library
# -----------------------
import argparse
import glob
import hashlib
import json

# -----------------------
# Logging
# -----------------------
import logging
import os
import shutil
import time
import warnings
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Any

# -----------------------
# Third-party
# -----------------------
import dask
import dask.dataframe as dd
import lsdb
import pandas as pd

# -----------------------
# Project
# -----------------------
from crossmatch_auto import crossmatch_auto
from crossmatch_cross import crossmatch_tiebreak_safe
from dask.distributed import Client, as_completed, performance_report
from dask.distributed import wait as dask_wait
from deduplication import (
    REPRESENTATIVE_RADIUS_DIAGNOSTIC_COLUMN,
    build_global_tie_invariant_diagnostics,
    count_global_edge_group_mismatches,
    count_global_tie_invariant_violations,
    filter_dask_by_tie_treatment,
    filter_pandas_by_tie_treatment,
    run_dedup_with_lsdb_map_partitions,
    validate_instrument_type_inclusion,
    validate_object_type_inclusion,
    validate_spatial_safety,
)
from executor import get_executor
from product_handle import save_dataframe
from resource_usage import ResourceUsageMonitor
from specz import (
    HOMOGENIZED_COLUMNS,
    build_runtime_schema_hints,
    prepare_catalog,
    validate_combine_configuration,
)
from specz_homogenization import validate_translation_config
from utils import (
    configure_exception_hook,
    configure_warning_handler,
    dump_yml,
    ensure_crc_logger,
    load_yml,
    log_step,
    read_completed_steps,
    start_crc_log_collector,
    update_process_info,
)
from worker_health import WorkerFloorMonitor

__all__ = ["main"]

_resource_usage_monitor: ResourceUsageMonitor | None = None
_worker_floor_monitor: WorkerFloorMonitor | None = None

DIAGNOSTICS_DEFAULTS = {
    "tie_invariant_diagnostics_enabled": True,
    "tie_invariant_diagnostics_detailed_enabled": False,
    "tie_invariant_diagnostics_sample_size": 10,
    "tie_invariant_diagnostics_max_rows": 100,
    "label_merge_diagnostics_enabled": True,
    "crossmatch_geometry_diagnostics_enabled": False,
    "representative_radius_diagnostics_enabled": False,
    "dedup_edge_diagnostics_enabled": False,
    "save_expr_columns": False,
}

PREPARATION_DEFAULTS = {
    "repartition_prepared_catalogs": False,
    "prepared_partition_size": "256MB",
}

TRANSLATION_FALLBACK_DEFAULTS = {
    "crossmatch_radius_arcsec": 0.5,
    "max_representative_radius_arcsec": None,
    "margin_threshold_arcsec": 5.0,
    "margin_warning_fraction": 0.8,
    "crossmatch_n_neighbors": 160,
    "crossmatch_saturation_enabled": False,
    "crossmatch_saturation_warn_fraction": 0.01,
    "crossmatch_saturation_fail_fraction": None,
    "delta_z_threshold": 0.0,
    "validate_global_graph_edges": False,
    "validate_global_tie_invariants": True,
    "validate_crd_id_uniqueness": True,
}

PUBLISH_MAX_ATTEMPTS = 3
PUBLISH_RETRY_DELAY_SECONDS = 5


def _build_runtime_param_config(param_config: dict | None) -> dict:
    """Build the internal runtime param mapping from the canonical config layout."""
    if param_config is None:
        return {}
    if not isinstance(param_config, dict):
        raise TypeError("param must be a mapping")

    allowed_sections = {"run", "filters", "preparation", "diagnostics", "output"}
    unknown_sections = sorted(set(param_config) - allowed_sections)
    if unknown_sections:
        raise ValueError(f"param has unknown section(s): {unknown_sections}")

    normalized = dict(param_config)

    run = normalized.get("run")
    if not isinstance(run, dict):
        raise TypeError("param.run must be a mapping")
    unknown_run = sorted(
        set(run) - {"combine_type", "tie_treatment_option", "flags_translation_file"}
    )
    if unknown_run:
        raise ValueError(f"param.run has unknown option(s): {unknown_run}")
    for key in ("combine_type", "tie_treatment_option", "flags_translation_file"):
        if key in run:
            normalized[key] = run[key]

    filters = normalized.get("filters")
    if not isinstance(filters, dict):
        raise TypeError("param.filters must be a mapping")
    unknown_filters = sorted(
        set(filters)
        - {
            "z_flag_homogenized_value_to_cut",
            "instrument_type_homogenized",
            "object_type_homogenized",
        }
    )
    if unknown_filters:
        raise ValueError(f"param.filters has unknown option(s): {unknown_filters}")
    if "z_flag_homogenized_value_to_cut" in filters:
        normalized["z_flag_homogenized_value_to_cut"] = filters[
            "z_flag_homogenized_value_to_cut"
        ]

    instrument = filters.get("instrument_type_homogenized")
    if not isinstance(instrument, dict):
        raise TypeError("param.filters.instrument_type_homogenized must be a mapping")
    unknown_instrument = sorted(
        set(instrument)
        - {
            "include_spectroscopic",
            "include_grism",
            "include_photometric",
            "include_unclassified",
        }
    )
    if unknown_instrument:
        raise ValueError(
            "param.filters.instrument_type_homogenized has unknown option(s): "
            f"{unknown_instrument}"
        )
    instrument_aliases = {
        "include_spectroscopic": "include_spectroscopic_ith",
        "include_grism": "include_grism_ith",
        "include_photometric": "include_photometric_ith",
        "include_unclassified": "include_unclassified_ith",
    }
    for source, target in instrument_aliases.items():
        if source in instrument:
            normalized[target] = instrument[source]

    object_type = filters.get("object_type_homogenized")
    if not isinstance(object_type, dict):
        raise TypeError("param.filters.object_type_homogenized must be a mapping")
    unknown_object_type = sorted(
        set(object_type)
        - {
            "include_unclassified",
            "include_galaxy",
            "include_star",
            "include_agn",
            "include_qso",
            "include_galactic",
        }
    )
    if unknown_object_type:
        raise ValueError(
            "param.filters.object_type_homogenized has unknown option(s): "
            f"{unknown_object_type}"
        )
    object_aliases = {
        "include_unclassified": "include_unclassified_oth",
        "include_galaxy": "include_galaxy_oth",
        "include_star": "include_star_oth",
        "include_agn": "include_agn_oth",
        "include_qso": "include_qso_oth",
        "include_galactic": "include_galactic_oth",
    }
    for source, target in object_aliases.items():
        if source in object_type:
            normalized[target] = object_type[source]

    output = normalized.get("output")
    if not isinstance(output, dict):
        raise TypeError("param.output must be a mapping")
    unknown_output = sorted(
        set(output)
        - {
            "extra_columns",
            "homogenized_columns",
            "insert_DP1_footprint_flag",
            "insert_rubin_footprint_flag",
        }
    )
    if unknown_output:
        raise ValueError(f"param.output has unknown option(s): {unknown_output}")
    output_aliases = {
        "extra_columns": "extra_columns",
        "homogenized_columns": "output_homogenized_columns",
        "insert_DP1_footprint_flag": "insert_DP1_footprint_flag",
        "insert_rubin_footprint_flag": "insert_rubin_footprint_flag",
    }
    for source, target in output_aliases.items():
        if source in output:
            normalized[target] = output[source]

    return normalized


def _log_translation_fallbacks(
    translation_config: dict,
    logger: logging.LoggerAdapter | logging.Logger,
) -> None:
    """Warn when optional scientific settings rely on internal defaults."""
    missing = {
        key: default
        for key, default in TRANSLATION_FALLBACK_DEFAULTS.items()
        if key not in translation_config
    }
    if not missing:
        return
    logger.warning(
        "flags_translation.yaml is missing optional scientific setting(s); "
        "using internal fallback defaults: %s",
        missing,
    )


def _merge_param_diagnostics(
    translation_config: dict,
    param_config: dict,
) -> dict:
    """Overlay operational diagnostics from param.diagnostics onto runtime config."""
    merged = dict(translation_config or {})
    diagnostics = param_config.get("diagnostics") or {}
    if not isinstance(diagnostics, dict):
        raise TypeError("param.diagnostics must be a mapping")

    unknown = sorted(set(diagnostics) - set(DIAGNOSTICS_DEFAULTS))
    if unknown:
        raise ValueError(f"param.diagnostics has unknown option(s): {unknown}")

    for key, default in DIAGNOSTICS_DEFAULTS.items():
        merged[key] = diagnostics.get(key, merged.get(key, default))
    merged["expr_column_schema"] = merged.get("expr_column_schema") or {}

    for key in (
        "tie_invariant_diagnostics_enabled",
        "tie_invariant_diagnostics_detailed_enabled",
        "label_merge_diagnostics_enabled",
        "crossmatch_geometry_diagnostics_enabled",
        "representative_radius_diagnostics_enabled",
        "dedup_edge_diagnostics_enabled",
        "save_expr_columns",
    ):
        if not isinstance(merged[key], bool):
            raise TypeError(f"param.diagnostics.{key} must be a boolean")

    for key in (
        "tie_invariant_diagnostics_sample_size",
        "tie_invariant_diagnostics_max_rows",
    ):
        value = merged[key]
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"param.diagnostics.{key} must be a positive integer")

    if not isinstance(merged["expr_column_schema"], dict):
        raise TypeError("flags_translation.expr_column_schema must be a mapping")

    return merged


def _merge_param_preparation(
    translation_config: dict,
    param_config: dict,
) -> dict:
    """Overlay operational preparation settings from param.preparation."""
    merged = dict(translation_config or {})
    preparation = param_config.get("preparation") or {}
    if not isinstance(preparation, dict):
        raise TypeError("param.preparation must be a mapping")

    unknown = sorted(set(preparation) - set(PREPARATION_DEFAULTS))
    if unknown:
        raise ValueError(f"param.preparation has unknown option(s): {unknown}")

    for key, default in PREPARATION_DEFAULTS.items():
        merged[key] = preparation.get(key, merged.get(key, default))

    if not isinstance(merged["repartition_prepared_catalogs"], bool):
        raise TypeError("param.preparation.repartition_prepared_catalogs must be a boolean")
    if not isinstance(merged["prepared_partition_size"], str):
        raise TypeError("param.preparation.prepared_partition_size must be a string")

    return merged


# -----------------------
# Helpers
# -----------------------
def _log_remote_future_exception(
    lg: logging.LoggerAdapter, fut, msg_prefix: str, extra: dict | None = None
) -> None:
    """Log the remote (worker-side) traceback carried by a Dask Future."""
    import traceback as _tb

    try:
        err = fut.exception()
        tb = fut.traceback()  # remote TB; can be None
        if tb is not None:
            lg.error(
                "%s: %r", msg_prefix, err, exc_info=(type(err), err, tb), extra=extra
            )
        else:
            # Fallback: driver-side traceback (should still exist)
            lg.error("%s: %r\n%s", msg_prefix, err, _tb.format_exc(), extra=extra)
    except Exception:
        lg.error(
            "%s (and failed to render remote traceback)\n%s",
            msg_prefix,
            _tb.format_exc(),
            extra=extra,
        )


def _release_dask_futures(
    futures: list[Any],
    lg: logging.LoggerAdapter,
    *,
    client: Client | None = None,
    cancel: bool = False,
    context: str,
) -> None:
    """Drop scheduler references to submitted futures.

    Dask may recompute a completed Future if the worker holding its result dies
    and the client still references it. Preparation writes files as a side
    effect, so those futures must not remain live after their result is copied.
    """
    unique_futures = list(dict.fromkeys(futures))
    if not unique_futures:
        return

    if cancel and client is not None:
        try:
            client.cancel(unique_futures, force=True)
        except Exception as e:
            lg.debug("Could not cancel %s futures: %s", context, e)

    released = 0
    for fut in unique_futures:
        try:
            fut.release()
            released += 1
        except Exception as e:
            lg.debug("Could not release %s future %r: %s", context, fut, e)

    lg.info("Released %d Dask future(s) for %s.", released, context)


def _cancel_local_futures(
    futures: list[Any], lg: logging.LoggerAdapter, *, context: str
) -> None:
    """Best-effort cancellation of local ThreadPoolExecutor futures."""
    pending = [fut for fut in futures if not fut.done()]
    if not pending:
        return

    cancelled = 0
    for fut in pending:
        try:
            if fut.cancel():
                cancelled += 1
        except Exception as e:
            lg.debug("Could not cancel %s future %r: %s", context, fut, e)

    lg.warning(
        "%s failed; requested cancellation for %d local future(s), "
        "%d accepted cancellation. Running futures will finish before shutdown.",
        context,
        len(pending),
        cancelled,
    )


def _remove_artifact_if_exists(
    path: str, lg: logging.LoggerAdapter, *, label: str
) -> None:
    """Remove a possibly partial file or directory left by a failed phase."""
    if not path:
        return
    try:
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
            lg.warning("Removed partial %s directory: %s", label, path)
        elif os.path.isfile(path):
            os.remove(path)
            lg.warning("Removed partial %s file: %s", label, path)
    except Exception as e:
        lg.warning("Could not remove partial %s at %s: %s", label, path, e)


def _phase_logger(base_logger: logging.Logger, phase: str) -> logging.LoggerAdapter:
    """Return a LoggerAdapter that injects the phase into records."""
    return logging.LoggerAdapter(base_logger, {"phase": phase})


def _filesize_mb(path: str) -> float:
    """Return file or directory tree size in MB. On error, inf."""
    try:
        if os.path.isdir(path):
            total_size = 0
            for root, _, files in os.walk(path):
                for filename in files:
                    total_size += os.path.getsize(os.path.join(root, filename))
            return total_size / 1024 / 1024
        return os.path.getsize(path) / 1024 / 1024
    except Exception:
        return float("inf")


def _ensure_non_empty_final_dataframe(
    df,
    logger: logging.LoggerAdapter,
    *,
    context: str,
) -> None:
    """Fail clearly if a Pandas or Dask final dataframe has no rows."""
    if _is_dask_dataframe(df):
        sample = df.head(1, npartitions=-1, compute=True)
        is_empty = len(sample) == 0
    else:
        is_empty = len(df) == 0

    if is_empty:
        msg = (
            "Final catalog is empty before export "
            f"({context}). Nothing can be consolidated."
        )
        logger.error(msg)
        raise RuntimeError(msg)


def _is_dask_dataframe(df: Any) -> bool:
    """Return True when df is a Dask DataFrame-like object."""
    return isinstance(df, dd.DataFrame) or (
        hasattr(df, "map_partitions") and hasattr(df, "to_parquet")
    )


def _is_collection_root(path: str) -> bool:
    """Return True if path contains collection.properties (HATS root)."""
    return (
        bool(path)
        and os.path.isdir(path)
        and os.path.exists(os.path.join(path, "collection.properties"))
    )


def _is_hats_subcatalog(path: str) -> bool:
    """Return True if path contains hats.properties (HATS subcatalog)."""
    return (
        bool(path)
        and os.path.isdir(path)
        and os.path.exists(os.path.join(path, "hats.properties"))
    )


def _is_hats_collection(path: str) -> bool:
    """Return True if root or subcatalog."""
    return _is_collection_root(path) or _is_hats_subcatalog(path)


def _normalize_collection_root(path: str | None) -> str | None:
    """Return normalized collection root given a root or subcatalog path."""
    if not path:
        return path
    p = path.rstrip("/")
    if _is_collection_root(p):
        return p
    if _is_hats_subcatalog(p):
        parent = os.path.dirname(p)
        return parent if _is_collection_root(parent) else p
    return p


def _guess_collection_for_step(temp_dir: str, step: int) -> str | None:
    """Heuristically discover the imported collection root for a given step."""
    candidates = [
        os.path.join(temp_dir, f"merged_step{step}_hats"),
        os.path.join(temp_dir, f"merged_step{step}.hats"),
    ]
    for cand in candidates:
        if _is_collection_root(cand):
            return cand

    patterns = [
        os.path.join(temp_dir, f"merged_step{step}", "*_hats"),
        os.path.join(temp_dir, f"merged_step{step}", "*.hats"),
    ]
    for pat in patterns:
        for cand in glob.glob(pat):
            cand_root = _normalize_collection_root(cand)
            if cand_root and _is_collection_root(cand_root):
                return cand_root

    for props in glob.glob(
        os.path.join(temp_dir, "**", "collection.properties"), recursive=True
    ):
        root = os.path.dirname(props)
        if f"step{step}" in root and _is_collection_root(root):
            return root

    return None


def _resume_set(
    resume_log_path: str, key: str, value: str, lg: logging.LoggerAdapter
) -> None:
    """Append a key/value checkpoint entry into the resume log."""
    try:
        data = {}
        if os.path.exists(resume_log_path):
            with open(resume_log_path, "r") as f:
                for line in f:
                    if line.startswith("{"):
                        d = json.loads(line)
                        data.update(d)
        data[key] = value
        with open(resume_log_path, "a") as f:
            f.write(json.dumps({key: value}) + "\n")
        lg.info("Resume checkpoint saved: %s = %s", key, value)
    except Exception as e:
        lg.warning("Could not update resume log: %s", e)


def _resume_get(resume_log_path: str, key: str) -> str | None:
    """Return last value for a key from the resume log."""
    try:
        if not os.path.exists(resume_log_path):
            return None
        val = None
        with open(resume_log_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or not line.startswith("{"):
                    continue
                d = json.loads(line)
                if key in d:
                    val = d[key]
        return val
    except Exception:
        return None


def _cleanup_previous_step(
    step_index: int,
    prepared_info: list[dict[str, Any]],
    temp_dir: str,
    lg: logging.LoggerAdapter,
) -> None:
    """Delete artifacts from *all* previous steps to save disk space.

    For every step k < step_index, removes:
      - prepared_<internal_name>
      - prepared_<internal_name>_hats
      - merged_step<k>
      - merged_step<k>_hats
    Also removes any recorded collection_path for previous prepared entries.
    Auto-crossmatch roots are intentionally kept until final cleanup so Dask
    retries can reuse them instead of recomputing expensive self-crossmatches.
    """

    def _rm_path(p: str) -> None:
        if not p:
            return
        try:
            if os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True)
            elif os.path.isfile(p):
                os.remove(p)
            else:
                return
            lg.info("Deleted artifact: %s", p)
        except Exception as e:
            lg.warning("Could not delete %s: %s", p, e)

    # 1) Remove all merged_step<k> and merged_step<k>_hats for k < step_index
    try:
        for entry in os.listdir(temp_dir):
            full = os.path.join(temp_dir, entry)
            if not os.path.isdir(full):
                continue

            # Accept both patterns:
            #   - merged_step<k>
            #   - merged_step<k>_hats
            name = entry.strip()
            if not name.startswith("merged_step"):
                continue

            tail = name.replace("merged_step", "", 1)
            # tail can be like "5" or "5_hats"
            num_str = tail.split("_", 1)[0].strip()
            try:
                k = int(num_str)
            except Exception:
                continue

            if k < step_index:
                _rm_path(full)
    except Exception as e:
        lg.warning("Could not list merged_step folders under %s: %s", temp_dir, e)

    # 2) Remove prepared artifacts for all previous prepared entries
    for i in range(0, min(step_index, len(prepared_info))):
        prev = prepared_info[i]

        # Base prepared_<internal_name>
        base_prepared = prev.get("prepared_path")
        if base_prepared:
            _rm_path(base_prepared)
            _rm_path(base_prepared + "_hats")

        # Recorded collection_path (may point to hats/auto variants)
        coll = prev.get("collection_path")
        if coll and not os.path.basename(os.path.normpath(coll)).endswith("_hats_auto"):
            _rm_path(coll)


# -----------------------
# Driver task for auto self-crossmatch
# -----------------------


def _run_auto_cross(
    info: dict, logs_dir: str, client: Client, translation_config: dict
):
    """Run one self-crossmatch from the driver.

    LSDB submits its internal Dask graph to the active distributed client. This
    function must not itself be submitted as a Dask task, otherwise its
    ``compute`` and ``write_catalog`` calls create nested scheduler submissions
    from a worker.
    """
    with client.as_current():
        hats_path = info["prepared_path"] + "_hats"
        if not os.path.isdir(hats_path):
            raise FileNotFoundError(
                f"Expected prepared collection not found: {hats_path}"
            )
        cat = lsdb.open_catalog(hats_path)
        return crossmatch_auto(
            catalog=cat,
            collection_path=hats_path,  # base; writes "<base>_auto"
            logs_dir=logs_dir,
            translation_config=translation_config,
        )


# -----------------------
# Driver task for one crossmatch pair
# -----------------------
def _run_crossmatch_pair(
    left_collection_path: str,
    right_collection_path: str,
    logs_dir: str,
    temp_dir: str,
    step: int,
    client: Client,
    translation_config: dict,
) -> str:
    """Open two collections and submit their LSDB graph from the driver."""
    with client.as_current():
        left_cat = lsdb.open_catalog(left_collection_path)
        right_cat = lsdb.open_catalog(right_collection_path)

        return crossmatch_tiebreak_safe(
            left_cat=left_cat,
            right_cat=right_cat,
            logs_dir=logs_dir,
            temp_dir=temp_dir,
            step=step,
            client=client,
            translation_config=translation_config,
            do_import=True,
        )


# -----------------------
# Cleanup for crossmatch tournment
# -----------------------
def _cleanup_inputs_of_merge(
    left_root: str,
    right_root: str,
    lg: logging.LoggerAdapter,
) -> None:
    """
    Remove only the two input collections that were just merged.
    Accepts either a collection root or a subcatalog and normalizes to root.
    Keeps prepared *_hats_auto roots until final cleanup; they are expensive to
    recompute and may be needed if Dask retries old tasks.
    """

    def _to_root(p: str) -> str:
        n = _normalize_collection_root(p) or p
        return n

    def _rm_root(p: str) -> None:
        if not p:
            return
        name = os.path.basename(os.path.normpath(p))
        if name.startswith("prepared_") and name.endswith("_hats_auto"):
            lg.info("Keeping auto-crossmatch collection for retry safety: %s", p)
            return
        try:
            if os.path.isdir(p) and _is_collection_root(p):
                shutil.rmtree(p, ignore_errors=True)
                lg.info("Deleted input collection: %s", p)
        except Exception as e:
            lg.warning("Could not delete input collection %s: %s", p, e)

    lroot = _to_root(left_root)
    rroot = _to_root(right_root)
    if lroot and rroot and (lroot != rroot):
        if _is_collection_root(lroot):
            _rm_root(lroot)
        if _is_collection_root(rroot):
            _rm_root(rroot)


# -----------------------
# Publish / copy helpers
# -----------------------
def _copy_file(src: str, dst: str, lg: logging.LoggerAdapter) -> None:
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    abs_src, abs_dst = os.path.abspath(src), os.path.abspath(dst)
    if abs_src == abs_dst:
        lg.info("Skip copy: source equals destination (%s).", abs_dst)
        return
    try:
        if os.path.exists(abs_dst):
            os.remove(abs_dst)
        os.link(abs_src, abs_dst)  # hardlink when on the same filesystem
        lg.info("Hardlinked: %s -> %s", abs_src, abs_dst)
    except Exception:
        shutil.copy2(abs_src, abs_dst)
        lg.info("Copied: %s -> %s", abs_src, abs_dst)


def _copy_tree(src_dir: str, dst_dir: str, lg: logging.LoggerAdapter) -> None:
    """Copy src_dir -> dst_dir recursively, overwriting if present."""
    if not os.path.isdir(src_dir):
        lg.warning("Source dir not found for copy: %s", src_dir)
        return
    os.makedirs(dst_dir, exist_ok=True)
    for root, dirs, files in os.walk(src_dir):
        rel = os.path.relpath(root, src_dir)
        tgt_root = os.path.join(dst_dir, rel) if rel != "." else dst_dir
        os.makedirs(tgt_root, exist_ok=True)
        for f in files:
            _copy_file(os.path.join(root, f), os.path.join(tgt_root, f), lg)


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_copied_file(src: str, dst: str) -> None:
    if not os.path.isfile(src):
        raise FileNotFoundError(f"source file is missing: {src}")
    if not os.path.isfile(dst):
        raise FileNotFoundError(f"destination file is missing: {dst}")

    src_size = os.path.getsize(src)
    dst_size = os.path.getsize(dst)
    if src_size != dst_size:
        raise RuntimeError(
            "copied file size mismatch: "
            f"{src} ({src_size} bytes) != {dst} ({dst_size} bytes)"
        )

    src_hash = _sha256_file(src)
    dst_hash = _sha256_file(dst)
    if src_hash != dst_hash:
        raise RuntimeError(
            f"copied file checksum mismatch: {src} ({src_hash}) != {dst} ({dst_hash})"
        )


def _verify_copied_tree(src_dir: str, dst_dir: str) -> tuple[int, int]:
    if not os.path.isdir(src_dir):
        raise FileNotFoundError(f"source directory is missing: {src_dir}")
    if not os.path.isdir(dst_dir):
        raise FileNotFoundError(f"destination directory is missing: {dst_dir}")

    n_dirs = 0
    n_files = 0
    for root, dirs, files in os.walk(src_dir):
        rel = os.path.relpath(root, src_dir)
        dst_root = os.path.join(dst_dir, rel) if rel != "." else dst_dir
        if not os.path.isdir(dst_root):
            raise FileNotFoundError(f"destination directory is missing: {dst_root}")
        n_dirs += 1

        for dirname in dirs:
            dst_subdir = os.path.join(dst_root, dirname)
            if not os.path.isdir(dst_subdir):
                raise FileNotFoundError(
                    f"destination directory is missing: {dst_subdir}"
                )

        for filename in files:
            src_file = os.path.join(root, filename)
            dst_file = os.path.join(dst_root, filename)
            _verify_copied_file(src_file, dst_file)
            n_files += 1

    return n_dirs, n_files


def _verify_publish_artifacts(
    artifacts: list[tuple[str, str, str]],
    lg: logging.LoggerAdapter,
) -> None:
    """Verify published artifacts against staged sources."""
    total_files = 0
    total_dirs = 0
    for kind, src, dst in artifacts:
        if kind == "file":
            _verify_copied_file(src, dst)
            total_files += 1
        elif kind == "tree":
            n_dirs, n_files = _verify_copied_tree(src, dst)
            total_dirs += n_dirs
            total_files += n_files
        else:
            raise ValueError(f"Unknown publish artifact kind: {kind}")

    lg.info(
        "Publish verification passed: %d file(s), %d directories verified.",
        total_files,
        total_dirs,
    )


def _copy_publish_artifact(
    kind: str, src: str, dst: str, lg: logging.LoggerAdapter
) -> None:
    if kind == "file":
        _copy_file(src, dst, lg)
    elif kind == "tree":
        _copy_tree(src, dst, lg)
    else:
        raise ValueError(f"Unknown publish artifact kind: {kind}")


def _copy_and_verify_publish_artifacts(
    artifacts: list[tuple[str, str, str]],
    lg: logging.LoggerAdapter,
    max_attempts: int = PUBLISH_MAX_ATTEMPTS,
    retry_delay_seconds: int = PUBLISH_RETRY_DELAY_SECONDS,
) -> None:
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            lg.info("Publish copy/verify attempt %d/%d.", attempt, max_attempts)
            for kind, src, dst in artifacts:
                _copy_publish_artifact(kind, src, dst, lg)
            _verify_publish_artifacts(artifacts, lg)
            return
        except Exception as e:
            last_error = e
            if attempt >= max_attempts:
                lg.error(
                    "FAILED publish verification after %d attempt(s): %s",
                    max_attempts,
                    e,
                )
                raise

            lg.warning(
                "Publish copy/verify attempt %d/%d failed: %s. Retrying in %d s.",
                attempt,
                max_attempts,
                e,
                retry_delay_seconds,
            )
            time.sleep(retry_delay_seconds)

    if last_error is not None:
        raise last_error


def _snapshot_shell_logs(
    src_dir: str, dst_dir: str, lg: logging.LoggerAdapter, max_size_mb: int = 100
) -> None:
    """Copy *.log/*.err/*.out (and *.N rotations) from src_dir to dst_dir."""
    os.makedirs(dst_dir, exist_ok=True)
    patterns = ["*.log", "*.log.*", "*.err", "*.err.*", "*.out", "*.out.*"]
    seen = set()
    for pat in patterns:
        for path in glob.glob(os.path.join(src_dir, pat)):
            if path in seen:
                continue
            seen.add(path)
            try:
                if os.path.getsize(path) > max_size_mb * 1024 * 1024:
                    lg.info("Skip (>%d MB): %s", max_size_mb, path)
                    continue
            except Exception:
                pass
            _copy_file(path, os.path.join(dst_dir, os.path.basename(path)), lg)


# -----------------------
# Core pipeline
# -----------------------


def main(
    config_path: str, cwd: str = ".", base_dir_override: str | None = None
) -> None:
    """Run the CRC pipeline end-to-end."""
    global _resource_usage_monitor, _worker_floor_monitor

    delete_temp_files = True  # set True to aggressively clean intermediates

    # --- Load config ---
    config = load_yml(config_path)
    param_config = _build_runtime_param_config(config.get("param", {}))
    if base_dir_override is None:
        raise ValueError("You must specify --base_dir via the command line.")
    base_dir = base_dir_override

    # --- Paths ---
    output_root_dir = config["output_root_dir"]
    output_dir = config["output_dir"]
    out_root_and_dir = os.path.join(output_root_dir, output_dir)
    output_name = config["output_name"]
    output_format = config.get("output_format", "parquet").lower()

    logs_dir = os.path.join(base_dir, "process_info")
    temp_dir = os.path.join(base_dir, "temp")
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    launch_dir = os.environ.get("CRC_LAUNCH_DIR", ".")
    launch_snap_dir = os.path.join(logs_dir, "launch_dir_files")
    print(f"Pipeline was called from: {launch_dir}")

    try:
        _copy_file(
            config_path,
            os.path.join(base_dir, "config.yaml"),
            logging.LoggerAdapter(logging.getLogger("crc"), {"phase": "init"}),
        )
    except Exception:
        pass

    # --- process.yml bookkeeping ---
    try:
        main_process_info = os.path.join(cwd, "process.yml")
        if os.path.exists(main_process_info):
            _copy_file(
                main_process_info,
                os.path.join(base_dir, "process.yml"),
                logging.LoggerAdapter(logging.getLogger("crc"), {"phase": "init"}),
            )
    except Exception:
        pass

    process_info_path = os.path.join(base_dir, "process.yml")
    if not os.path.exists(process_info_path):
        dump_yml(process_info_path, {})
    process_info = load_yml(process_info_path) or {}
    update_process_info(process_info, process_info_path, "status", "Failed")
    update_process_info(
        process_info, process_info_path, "start_time", str(pd.Timestamp.now())
    )

    # --- Logger & warnings (driver) ---
    base_logger = ensure_crc_logger(logs_dir)
    log_init = _phase_logger(base_logger, "init")

    # Optional collector on the driver if CRC_LOG_COLLECTOR is set
    collector_env = os.getenv("CRC_LOG_COLLECTOR", "").strip()
    if collector_env:
        try:
            host, port_str = collector_env.split(":")
            port = int(port_str)
            bind_host = "0.0.0.0" if host not in ("127.0.0.1", "localhost") else host
            start_crc_log_collector(host=bind_host, port=port)
            log_init.info("CRC log collector listening on udp://%s:%d", bind_host, port)
        except Exception as e:
            log_init.warning(
                "Failed to start CRC log collector from '%s': %s", collector_env, e
            )

    log_init.info("START init: pipeline bootstrap")
    cut_value = param_config.get("z_flag_homogenized_value_to_cut", 0)
    try:
        cut_value_numeric = float(cut_value) if cut_value is not None else None
    except (TypeError, ValueError):
        cut_value_numeric = None
    if cut_value_numeric == 0.0:
        log_init.info(
            "z_flag_homogenized cut disabled explicitly "
            "(z_flag_homogenized_value_to_cut=0)."
        )
    configure_warning_handler(base_logger)
    warnings.filterwarnings(
        "ignore",
        message=".*Sending large graph of size.*",
        category=UserWarning,
        module="distributed",
    )
    configure_exception_hook(base_logger, process_info, process_info_path)

    # --- Quick config sanity logs (minimal, non-invasive) ---
    try:
        n_inputs = len((config.get("inputs") or {}).get("specz") or [])
        sample_names = [
            e.get("internal_name")
            for e in (config["inputs"]["specz"][:3] if n_inputs else [])
        ]
        log_init.info(
            "Config sanity: output_root_dir=%s, output_dir=%s, output_name=%s, output_format=%s, specz_inputs=%d (sample=%s)",
            config.get("output_root_dir"),
            config.get("output_dir"),
            config.get("output_name"),
            config.get("output_format", "parquet"),
            n_inputs,
            ", ".join(sample_names),
        )
    except Exception:
        log_init.warning("Could not summarize config/inputs.", exc_info=True)

    # --- Translation file ---
    path_to_translation_file = param_config.get("flags_translation_file")
    if path_to_translation_file is None:
        log_init.error("Missing 'flags_translation_file' in config!")
        return
    # Resolve relative paths against the config file location (minimal safety)
    if not os.path.isabs(path_to_translation_file):
        path_to_translation_file = os.path.abspath(
            os.path.join(os.path.dirname(config_path), path_to_translation_file)
        )
    source_translation_file = path_to_translation_file
    source_exists = os.path.isfile(source_translation_file)
    source_size_kb = (
        os.path.getsize(source_translation_file) / 1024.0 if source_exists else 0.0
    )
    log_init.info(
        "flags_translation_file source=%s (exists=%s, size=%.1f KB)",
        source_translation_file,
        source_exists,
        source_size_kb,
    )
    if not source_exists:
        raise FileNotFoundError(
            f"flags_translation_file not found: {source_translation_file}"
        )

    # Stage translation file inside base_dir and always use that staged copy.
    path_to_translation_file = os.path.join(
        base_dir, os.path.basename(source_translation_file)
    )
    try:
        _copy_file(source_translation_file, path_to_translation_file, log_init)
    except Exception as e:
        log_init.error(
            "Failed to stage flags_translation_file into base_dir: %s",
            e,
            exc_info=True,
        )
        raise

    try:
        translation_config = load_yml(path_to_translation_file)
        validate_translation_config(translation_config)
        _log_translation_fallbacks(translation_config, log_init)
        translation_config = _merge_param_diagnostics(
            translation_config,
            param_config,
        )
        translation_config = _merge_param_preparation(
            translation_config,
            param_config,
        )
    except Exception as e:
        log_init.error("Failed to parse flags_translation_file: %s", e, exc_info=True)
        raise
    crossmatch_radius_arcsec = float(
        translation_config.get("crossmatch_radius_arcsec", 0.5)
    )
    margin_threshold_arcsec = float(
        translation_config.get("margin_threshold_arcsec", 5.0)
    )
    margin_warning_fraction = float(
        translation_config.get("margin_warning_fraction", 0.8)
    )
    configured_max_radius = translation_config.get(
        "max_representative_radius_arcsec", None
    )
    max_representative_radius_arcsec = (
        None if configured_max_radius is None else float(configured_max_radius)
    )
    validate_spatial_safety(
        crossmatch_radius_arcsec,
        margin_threshold_arcsec,
        margin_warning_fraction,
        max_representative_radius_arcsec,
    )
    log_init.info(
        'Spatial safety: crossmatch_radius=%.3f" margin_threshold=%.3f" '
        "max_representative_radius=%s ratio=%.3f",
        crossmatch_radius_arcsec,
        margin_threshold_arcsec,
        (
            "disabled"
            if max_representative_radius_arcsec is None
            else f'{max_representative_radius_arcsec:.3f}"'
        ),
        crossmatch_radius_arcsec / margin_threshold_arcsec,
    )
    # Minimal summary of loaded translation (no heavy dumping)
    try:
        tb = translation_config.get("tiebreaking_priority")
        log_init.info(
            "Loaded translation_config keys: %s",
            ", ".join(sorted(list(translation_config.keys()))[:8]),
        )
        if tb is not None:
            log_init.info("tiebreaking_priority=%s", tb)
        if "delta_z_threshold" in translation_config:
            log_init.info(
                "delta_z_threshold=%s", translation_config.get("delta_z_threshold")
            )
    except Exception:
        log_init.debug("Could not log translation_config summary.", exc_info=True)

    # --- Inputs sorted by size ---
    catalogs_unsorted = config["inputs"]["specz"]
    catalogs = sorted(catalogs_unsorted, key=lambda e: _filesize_mb(e["path"]))
    log_init.info("Catalogs sorted by disk size:")
    for entry in catalogs:
        log_init.info(
            " - %s: %.1f MB", entry["internal_name"], _filesize_mb(entry["path"])
        )

    combine_mode, validated_priorities = validate_combine_configuration(
        param_config.get("combine_type", "concatenate_and_mark_duplicates"),
        translation_config.get("tiebreaking_priority", []),
        param_config.get("z_flag_homogenized_value_to_cut", 0),
        log_init,
    )
    translation_config["tiebreaking_priority"] = validated_priorities
    instrument_type_inclusion = validate_instrument_type_inclusion(
        {
            key: param_config[key]
            for key in (
                "include_spectroscopic_ith",
                "include_grism_ith",
                "include_photometric_ith",
                "include_unclassified_ith",
            )
            if key in param_config
        }
    )
    object_type_inclusion = validate_object_type_inclusion(
        {
            key: param_config[key]
            for key in (
                "include_unclassified_oth",
                "include_galaxy_oth",
                "include_star_oth",
                "include_agn_oth",
                "include_qso_oth",
                "include_galactic_oth",
            )
            if key in param_config
        }
    )
    log_init.info("Instrument-type input filter inclusion: %s", instrument_type_inclusion)
    log_init.info("Object-type input filter inclusion: %s", object_type_inclusion)
    required_homogenized_columns: dict[str, list[str]] = {
        column: ["tiebreaking_priority"]
        for column in HOMOGENIZED_COLUMNS
        if column in set(validated_priorities)
    }
    if cut_value_numeric in {1.0, 2.0, 3.0, 4.0}:
        required_homogenized_columns.setdefault("z_flag_homogenized", []).append(
            "z_flag_homogenized_value_to_cut"
        )
    if not all(instrument_type_inclusion.values()):
        required_homogenized_columns.setdefault(
            "instrument_type_homogenized", []
        ).append("instrument_type_filter")
    if not all(object_type_inclusion.values()):
        required_homogenized_columns.setdefault(
            "object_type_homogenized", []
        ).append("object_type_filter")
    log_init.info("Required homogenized columns: %s", required_homogenized_columns)
    translation_config["runtime_schema_hints"] = build_runtime_schema_hints(
        param_config, translation_config
    )
    completed = read_completed_steps(os.path.join(temp_dir, "process_resume.log"))

    # --- Dask cluster/client ---
    cluster = get_executor(config["executor"], logs_dir=logs_dir)
    client = Client(cluster)
    _resource_usage_monitor = ResourceUsageMonitor(client)
    _resource_usage_monitor.start()

    # Do not start pipeline work until the configured worker floor is ready.
    exec_args = config.get("executor", {}).get("args", {}) or {}
    instance_cfg = exec_args.get("instance", {}) or {}
    scale_cfg = exec_args.get("scale", {}) or {}

    procs = int(instance_cfg.get("processes", 1) or 1)
    min_jobs = scale_cfg.get("minimum_jobs")
    min_workers = 0 if min_jobs is None else int(min_jobs) * procs
    recovery_timeout = float(
        scale_cfg.get("worker_recovery_timeout_seconds", 600.0)
    )
    recovery_interval = float(
        scale_cfg.get("worker_recovery_check_interval_seconds", 10.0)
    )

    def _abort_degraded_cluster() -> None:
        try:
            client.close(timeout=5)
        finally:
            cluster.close()

    if min_workers > 0:
        log_init.info(
            "Waiting up to %.1fs for minimum_workers=%d to start...",
            recovery_timeout,
            min_workers,
        )
        try:
            client.wait_for_workers(min_workers, timeout=recovery_timeout)
        except Exception as exc:
            try:
                current_workers = len(client.scheduler_info().get("workers", {}))
            except Exception:
                current_workers = -1
            log_init.error(
                "Minimum worker floor was not reached during startup: "
                "minimum_workers=%d current=%d timeout=%.1fs. Aborting.",
                min_workers,
                current_workers,
                recovery_timeout,
            )
            try:
                _abort_degraded_cluster()
            except Exception:
                log_init.exception("Failed to close cluster after startup timeout.")
            raise RuntimeError(
                "Dask minimum worker floor was not reached during startup."
            ) from exc
        else:
            current_workers = len(client.scheduler_info().get("workers", {}))
            log_init.info(
                "Confirmed: minimum_workers=%d started within %.1fs (current=%d).",
                min_workers,
                recovery_timeout,
                current_workers,
            )

    current_workers = len(client.scheduler_info().get("workers", {}))
    log_init.info(
        "WORKERS STILL RUNNING=%d.",
        current_workers,
    )

    if min_workers > 0:
        _worker_floor_monitor = WorkerFloorMonitor(
            client,
            minimum_workers=min_workers,
            recovery_timeout_seconds=recovery_timeout,
            check_interval_seconds=recovery_interval,
            logger=_phase_logger(base_logger, "resources"),
            on_timeout=_abort_degraded_cluster,
        )
        _worker_floor_monitor.start()
        log_init.info(
            "Worker-floor monitor enabled: minimum_workers=%d "
            "recovery_timeout=%.1fs check_interval=%.1fs.",
            min_workers,
            recovery_timeout,
            recovery_interval,
        )

    log_init.info("END init: pipeline bootstrap")

    # Dask perf report (global)
    global_report_path = os.path.join(logs_dir, "main_dask_report.html")
    with performance_report(filename=global_report_path):
        # -----------------------
        # 1) PREPARATION
        # -----------------------
        log_prep = _phase_logger(base_logger, "preparation")
        log_prep.info(
            "START preparation: reading inputs and building prepared collections (temp=%s)",
            temp_dir,
        )

        max_inflight = int(param_config.get("prepare_max_inflight", 5))

        def _prebuilt_result_tuple(entry: dict) -> tuple[str, str, str, str, str]:
            base = os.path.join(temp_dir, f"prepared_{entry['internal_name']}")
            empty_marker = f"{base}.empty"
            if os.path.exists(empty_marker):
                log_prep.warning(
                    "Skipping previously prepared empty catalog %s; marker=%s",
                    entry["internal_name"],
                    empty_marker,
                )
                return ("", "ra", "dec", entry["internal_name"], "empty_after_cut")
            hats = f"{base}_hats"
            if os.path.isdir(hats):
                return (hats, "ra", "dec", entry["internal_name"], "")
            hats_auto = f"{hats}_auto"
            if os.path.isdir(hats_auto):
                return (hats_auto, "ra", "dec", entry["internal_name"], "")
            log_prep.warning(
                "Prepared collection not found for %s. Expected %s or %s",
                entry["internal_name"],
                hats,
                hats_auto,
            )
            return (hats, "ra", "dec", entry["internal_name"], "")

        prepare_futures: list[Any] = []

        def _submit_prepare(entry: dict):
            fut = client.submit(
                prepare_catalog,
                entry,
                translation_config,
                param_config,
                logs_dir,
                temp_dir,
                combine_mode,
                pure=False,
            )
            prepare_futures.append(fut)
            return fut

        queue: list[dict] = []
        results: list[tuple[str, str, str, str, str]] = []
        for entry in catalogs:
            tag = f"prepare_{entry['internal_name']}"
            if tag in completed:
                log_prep.info("Skip already prepared: %s", entry["internal_name"])
                results.append(_prebuilt_result_tuple(entry))
            else:
                queue.append(entry)

        log_prep.info(
            "Concurrency for prepare: max_inflight=%d, to_prepare=%d, already_prepared=%d",
            max_inflight,
            len(queue),
            len(results),
        )

        ac = as_completed()
        inflight = []
        prepared_count = 0
        try:
            for _ in range(min(max_inflight, len(queue))):
                fut = _submit_prepare(queue.pop(0))
                ac.add(fut)
                inflight.append(fut)

            while inflight:
                fut = next(ac)
                inflight.remove(fut)
                try:
                    out = fut.result()
                except Exception:
                    # Logs remote worker traceback and repropagates
                    _log_remote_future_exception(
                        log_prep,
                        fut,
                        msg_prefix="PREPARE FAILED (remote worker traceback)",
                    )
                    raise
                results.append(out)
                _release_dask_futures(
                    [fut],
                    log_prep,
                    context=f"completed preparation {out[3]}",
                )
                prepared_count += 1
                if queue:
                    fut2 = _submit_prepare(queue.pop(0))
                    ac.add(fut2)
                    inflight.append(fut2)
        except Exception:
            _release_dask_futures(
                inflight,
                log_prep,
                client=client,
                cancel=True,
                context="in-flight preparation after failure",
            )
            _release_dask_futures(
                prepare_futures,
                log_prep,
                context="submitted preparation after failure",
            )
            raise
        finally:
            prepare_futures.clear()

        log_prep.info(
            "Prepared %d new catalogs; total results now %d / %d",
            prepared_count,
            len(results),
            len(catalogs),
        )
        if len(results) != len(catalogs):
            raise RuntimeError("Internal error: prepared results count mismatch.")

        empty_results = [r for r in results if r[4]]
        if empty_results:
            for r in empty_results:
                log_prep.warning(
                    "Catalog %s was excluded from subsequent steps because "
                    "preparation returned status=%s.",
                    r[3],
                    r[4],
                )
            log_prep.warning(
                "Excluded %d empty catalog(s) after preparation; continuing with "
                "%d catalog(s).",
                len(empty_results),
                len(results) - len(empty_results),
            )

        homogenized_metadata = []
        for r in results:
            metadata_path = os.path.join(
                temp_dir, f"prepared_{r[3]}.homogenized.json"
            )
            try:
                with open(metadata_path, encoding="utf-8") as fp:
                    homogenized_metadata.append(json.load(fp))
            except FileNotFoundError:
                log_prep.warning(
                    "Missing homogenized metadata for catalog %s at %s. "
                    "Required-column global validation will count it as empty.",
                    r[3],
                    metadata_path,
                )
                homogenized_metadata.append(
                    {
                        "product": r[3],
                        "homogenized_non_null_counts": {},
                    }
                )
            except Exception as e:
                raise RuntimeError(
                    f"Could not read homogenized metadata for catalog {r[3]} "
                    f"at {metadata_path}: {e}"
                ) from e

        for column, reasons in required_homogenized_columns.items():
            per_catalog_counts = {
                str(meta.get("product") or "unknown"): int(
                    (meta.get("homogenized_non_null_counts") or {}).get(column, 0)
                )
                for meta in homogenized_metadata
            }
            total_non_null = sum(per_catalog_counts.values())
            if total_non_null == 0:
                raise RuntimeError(
                    f"All input catalogs have no valid values for required "
                    f"homogenized column '{column}'. Required because: {reasons}. "
                    f"Per-catalog non-null counts: {per_catalog_counts}. "
                    "Check input column mappings and flags_translation.yaml."
                )
            empty_catalogs_for_column = [
                name for name, count in per_catalog_counts.items() if count == 0
            ]
            if empty_catalogs_for_column:
                log_prep.warning(
                    "Required homogenized column %s has no non-null values in "
                    "some input catalogs: %s; continuing because other catalogs "
                    "provide valid values. Reasons=%s",
                    column,
                    empty_catalogs_for_column,
                    reasons,
                )

        prepared_info = [
            {
                "collection_path": r[0],
                "prepared_path": os.path.join(temp_dir, f"prepared_{r[3]}"),
                "ra": r[1],
                "dec": r[2],
                "internal_name": r[3],
            }
            for r in results
            if not r[4]
        ]

        if not prepared_info:
            if len(empty_results) == len(results) and all(
                r[4] in {"empty_after_cut", "empty_after_filter"}
                for r in empty_results
            ):
                names = ", ".join(f"{r[3]} ({r[4]})" for r in empty_results)
                raise RuntimeError(
                    "All input catalogs became empty after applying preparation "
                    "filters; no non-empty catalogs remain for downstream "
                    f"processing. Empty catalogs: {names}."
                )
            raise RuntimeError(
                "All input catalogs were excluded during preparation; no non-empty "
                "catalogs remain for downstream processing."
            )

        # Mark prepares as done
        resume_log = os.path.join(temp_dir, "process_resume.log")
        for entry in catalogs:
            tag = f"prepare_{entry['internal_name']}"
            if tag not in completed:
                log_step(resume_log, tag)

        current_workers = len(client.scheduler_info().get("workers", {}))
        log_prep.info(
            "WORKERS STILL RUNNING=%d.",
            current_workers,
        )

        log_prep.info("END preparation: finished prepared collections")

        # -----------------------
        # Detect pipeline state for resume (is crossmatch already done?)
        # -----------------------
        final_step = len(prepared_info) - 1
        resume_log = os.path.join(temp_dir, "process_resume.log")

        def _recover_final_collection_path() -> str | None:
            # 1) Try to take from the resume of the last step
            resumed = _resume_get(
                resume_log, f"crossmatch_step{final_step}.collection_path"
            )
            resumed = _normalize_collection_root(resumed)
            if resumed and _is_collection_root(resumed):
                return resumed
            # 2) Try to guess on the disk
            guessed = _guess_collection_for_step(temp_dir, final_step)
            if guessed and _is_collection_root(guessed):
                return guessed
            return None

        merged_final_path = os.path.join(temp_dir, f"merged_step{final_step}")
        crossmatch_already_done = (
            f"crossmatch_step{final_step}" in completed
        ) or os.path.isdir(merged_final_path)

        # If you have already finished crossmatching, retrieve the collection root (if it exists) for distributed dedup.
        final_collection_path = (
            _recover_final_collection_path() if crossmatch_already_done else None
        )

        # -----------------------
        # 2) AUTO MATCH (self crossmatch over each prepared)
        # -----------------------
        log_auto = _phase_logger(base_logger, "automatch")
        if combine_mode in (
            "concatenate_and_mark_duplicates",
            "concatenate_and_remove_duplicates",
        ):
            try:
                if crossmatch_already_done:
                    # No auto needed: merges already completed and merged_step{final} present
                    log_auto.info(
                        "Skip automatch: crossmatch already finalized (merged_step%d exists or logged).",
                        final_step,
                    )
                else:
                    log_auto.info(
                        "START automatch: generating *_hats_auto from prepared collections"
                    )

                    queue_auto: list[dict] = []
                    already_done = 0
                    for info in prepared_info:
                        tag = f"autocross_{info['internal_name']}"
                        hats_auto = info["prepared_path"] + "_hats_auto"
                        if tag in completed and os.path.isdir(hats_auto):
                            already_done += 1
                        else:
                            queue_auto.append(info)

                    max_inflight_auto = int(
                        param_config.get("auto_cross_max_inflight", 2)
                    )
                    if max_inflight_auto < 1:
                        raise ValueError("auto_cross_max_inflight must be at least 1")
                    log_auto.info(
                        "Concurrency for auto-cross: max_inflight=%d, to_run=%d, already_done=%d",
                        max_inflight_auto,
                        len(queue_auto),
                        already_done,
                    )

                    if queue_auto:
                        auto_done_now = 0

                        def _submit_auto(executor: ThreadPoolExecutor, i: dict):
                            """Start one driver-side LSDB submission."""
                            log_auto.info(
                                "Submitting AUTO-CROSS for %s | prepared_path=%s",
                                i.get("internal_name"),
                                i.get("prepared_path"),
                            )
                            return executor.submit(
                                _run_auto_cross,
                                i,
                                logs_dir,
                                client,
                                translation_config,
                            )

                        with ThreadPoolExecutor(
                            max_workers=max_inflight_auto,
                            thread_name_prefix="crc-auto",
                        ) as auto_executor:
                            inflight2: dict[Any, dict] = {}

                            for _ in range(min(max_inflight_auto, len(queue_auto))):
                                info = queue_auto.pop(0)
                                inflight2[_submit_auto(auto_executor, info)] = info

                            while inflight2:
                                done, _ = wait(inflight2, return_when=FIRST_COMPLETED)
                                for fut in done:
                                    info_ctx = inflight2.pop(fut)
                                    name = info_ctx.get("internal_name", "<unknown>")
                                    prepared_path = info_ctx.get(
                                        "prepared_path", "<unknown>"
                                    )
                                    try:
                                        out_path = fut.result()
                                    except Exception:
                                        active = [
                                            ctx.get("internal_name", "<unknown>")
                                            for ctx in inflight2.values()
                                        ]
                                        log_auto.exception(
                                            "AUTO-CROSS FAILED for %s | prepared_path=%s | "
                                            "other_inflight=%s",
                                            name,
                                            prepared_path,
                                            active,
                                        )
                                        _cancel_local_futures(
                                            list(inflight2.keys()),
                                            log_auto,
                                            context="auto-cross",
                                        )
                                        _remove_artifact_if_exists(
                                            f"{prepared_path}_hats_auto",
                                            log_auto,
                                            label=f"auto-cross output for {name}",
                                        )
                                        raise

                                    info_ctx["collection_path"] = out_path
                                    log_auto.info(
                                        "AUTO-CROSS OK: %s | prepared_path=%s | output=%s",
                                        name,
                                        prepared_path,
                                        out_path,
                                    )
                                    log_step(resume_log, f"autocross_{name}")
                                    auto_done_now += 1

                                    if queue_auto:
                                        nxt = queue_auto.pop(0)
                                        inflight2[_submit_auto(auto_executor, nxt)] = (
                                            nxt
                                        )

                        log_auto.info(
                            "Auto crossmatch completed for %d catalogs (re/computed)",
                            auto_done_now,
                        )
                    else:
                        log_auto.info(
                            "No auto-crossmatch needed (all *_hats_auto present)."
                        )

                    current_workers = len(client.scheduler_info().get("workers", {}))
                    log_auto.info(
                        "WORKERS STILL RUNNING=%d.",
                        current_workers,
                    )

                    log_auto.info("END automatch: all *_hats_auto guaranteed on disk")

                # -----------------------
                # Enforcement of *_hats_auto ONLY if we are still going to run crossmatch
                # -----------------------
                if not crossmatch_already_done:
                    missing_auto: list[str] = []
                    for info in prepared_info:
                        hats_auto = info["prepared_path"] + "_hats_auto"
                        if os.path.isdir(hats_auto):
                            info["collection_path"] = hats_auto
                        else:
                            missing_auto.append(info["internal_name"])

                    if missing_auto:
                        # Detailed diagnostic before raising
                        log_auto.error(
                            "AUTO-CROSS VERIFICATION FAILED. Missing %d outputs. Expected *_hats_auto roots:",
                            len(missing_auto),
                        )
                        for info in prepared_info:
                            hats_auto = info["prepared_path"] + "_hats_auto"
                            log_auto.error(
                                " - %s | expected=%s | exists=%s",
                                info["internal_name"],
                                hats_auto,
                                os.path.isdir(hats_auto),
                            )

                        raise FileNotFoundError(
                            "Auto-cross outputs still missing after recovery: "
                            + ", ".join(missing_auto)
                            + ". Check disk/permissions/logs."
                        )

            except Exception:
                # Catch ANY driver-side error in the automatch phase
                log_auto.exception("Automatch FAILED with an unhandled error")
                raise

        # -----------------------
        # 3) CROSSMATCH (parallel tournament; no per-step resume)
        # -----------------------
        log_cross = _phase_logger(base_logger, "crossmatch")
        log_cross.info(
            "START crossmatch (parallel tournament) over prepared *_hats_auto"
        )

        if combine_mode == "concatenate":
            # Simple concat mode still passes through consolidation phase later.
            log_cross.info("Concatenate mode selected (no crossmatch).")
            try:
                log_cross.info(
                    "Concatenating %d prepared catalogs.", len(prepared_info)
                )
                for _pi in prepared_info[:5]:
                    log_cross.debug(
                        "Prepared path candidate: %s", _pi.get("prepared_path")
                    )
                df_final = dd.concat(
                    [dd.read_parquet(i["prepared_path"]) for i in prepared_info]
                )
                if output_format in {"hats", "parquet", "csv"}:
                    log_cross.info(
                        "Concatenate graph kept lazy for distributed output "
                        "format=%s "
                        "(npartitions=%s).",
                        output_format,
                        getattr(df_final, "npartitions", "unknown"),
                    )
                else:
                    df_final = df_final.compute()
                    log_cross.info(
                        "Concatenate compute finished: shape=%s",
                        tuple(df_final.shape),
                    )
            except Exception as e:
                import traceback as _tb

                log_cross.error(
                    "FAILED while concatenating prepared catalogs: %s\n%s",
                    repr(e),
                    _tb.format_exc(),
                )
                raise

            try:
                current_workers = len(client.scheduler_info().get("workers", {}))
                log_cross.info("WORKERS STILL RUNNING=%d.", current_workers)
            except Exception as e:
                log_cross.debug("Could not query scheduler_info workers: %s", e)

            log_cross.info("END crossmatch: concatenate mode (no crossmatch performed)")

        elif combine_mode in (
            "concatenate_and_mark_duplicates",
            "concatenate_and_remove_duplicates",
        ):
            if crossmatch_already_done:
                # Crossmatch chain already finished previously (by merged_step{final} presence)
                log_cross.info(
                    "Skip crossmatch: already finalized at step %d (merged_step present).",
                    final_step,
                )

            else:
                # === Parallel tournament without per-step resume ===
                try:
                    _resume_set(resume_log, "phase.crossmatch", "started", log_cross)

                    # Ensure all prepared_info[i]["collection_path"] are *_hats_auto
                    for info in prepared_info:
                        hats_auto = info["prepared_path"] + "_hats_auto"
                        if os.path.isdir(hats_auto):
                            info["collection_path"] = hats_auto
                        else:
                            raise FileNotFoundError(
                                f"Missing *_hats_auto for: {info['internal_name']}"
                            )

                    max_inflight_pairs = int(param_config.get("cross_max_inflight", 2))
                    if max_inflight_pairs < 1:
                        raise ValueError("cross_max_inflight must be at least 1")
                    log_cross.info(
                        "Max inflight crossmatch pairs: %d", max_inflight_pairs
                    )

                    # Build initial queue of nodes (label, root_path)
                    from collections import deque

                    nodes = deque()
                    for info in prepared_info:
                        root = (
                            _normalize_collection_root(info["collection_path"])
                            or info["collection_path"]
                        )
                        if not _is_hats_collection(root):
                            raise RuntimeError(f"Not a HATS collection: {root}")
                        root = (
                            _normalize_collection_root(root) or root
                        )  # normalize to root if subcatalog
                        nodes.append((info["internal_name"], root))

                    # Edge case: only one catalog
                    if len(nodes) == 1:
                        final_collection_path = nodes[0][1]
                        _resume_set(
                            resume_log, "phase.crossmatch", "completed", log_cross
                        )
                        log_cross.info(
                            "Only one prepared collection; skipping crossmatch. final=%s",
                            final_collection_path,
                        )
                    else:
                        # Driver-side futures pipeline. Each thread submits the
                        # LSDB graph directly to the distributed scheduler.
                        ready = deque(nodes)
                        carry: tuple[str, str] | None = None

                        inflight_meta: dict[Any, tuple] = {}
                        step_id = 1  # diagnostic numbering only

                        def _submit_pair(executor: ThreadPoolExecutor) -> bool:
                            nonlocal step_id
                            if len(ready) < 2:
                                return False
                            if len(inflight_meta) >= max_inflight_pairs:
                                return False
                            l_lab, l_path = ready.popleft()
                            r_lab, r_path = ready.popleft()
                            fut = executor.submit(
                                _run_crossmatch_pair,
                                l_path,
                                r_path,
                                logs_dir,
                                temp_dir,
                                step_id,
                                client,
                                translation_config,
                            )
                            inflight_meta[fut] = (
                                l_lab,
                                l_path,
                                r_lab,
                                r_path,
                                step_id,
                            )
                            log_cross.info(
                                "Submitted crossmatch pair: step=%d | %s VS %s",
                                step_id,
                                l_lab,
                                r_lab,
                            )
                            step_id += 1
                            return True

                        final_collection_path: str | None = None
                        with ThreadPoolExecutor(
                            max_workers=max_inflight_pairs,
                            thread_name_prefix="crc-cross",
                        ) as cross_executor:
                            while _submit_pair(cross_executor):
                                pass

                            while inflight_meta or (
                                len(ready) + (1 if carry else 0) > 1
                            ):
                                if inflight_meta:
                                    done, _ = wait(
                                        inflight_meta, return_when=FIRST_COMPLETED
                                    )
                                    for fut in done:
                                        meta = inflight_meta.pop(fut)
                                        l_lab, l_path, r_lab, r_path, sid = meta
                                        try:
                                            raw_out = fut.result()
                                        except Exception:
                                            active = [
                                                {
                                                    "step": m[4],
                                                    "left": m[0],
                                                    "right": m[2],
                                                }
                                                for m in inflight_meta.values()
                                            ]
                                            log_cross.exception(
                                                "Crossmatch failed (step %d: %s vs %s) | "
                                                "other_inflight=%s",
                                                sid,
                                                l_lab,
                                                r_lab,
                                                active,
                                            )
                                            _cancel_local_futures(
                                                list(inflight_meta.keys()),
                                                log_cross,
                                                context="crossmatch tournament",
                                            )
                                            for suffix in ("", "_hats", ".hats"):
                                                _remove_artifact_if_exists(
                                                    os.path.join(
                                                        temp_dir,
                                                        f"merged_step{sid}{suffix}",
                                                    ),
                                                    log_cross,
                                                    label=f"crossmatch step {sid} output",
                                                )
                                            _resume_set(
                                                resume_log,
                                                "phase.crossmatch",
                                                "failed",
                                                log_cross,
                                            )
                                            raise

                                        out_root = (
                                            _normalize_collection_root(raw_out)
                                            or raw_out
                                        )
                                        if not _is_hats_collection(out_root):
                                            log_cross.warning(
                                                "crossmatch returned a non-HATS path (step %d): %s",
                                                sid,
                                                raw_out,
                                            )

                                        if delete_temp_files:
                                            try:
                                                _cleanup_inputs_of_merge(
                                                    l_path, r_path, log_cross
                                                )
                                            except Exception as e:
                                                log_cross.warning(
                                                    "Cleanup of inputs for step %d failed (non-fatal): %s",
                                                    sid,
                                                    e,
                                                )

                                        ready.append((f"merged_step{sid}", out_root))
                                        final_collection_path = out_root

                                    while _submit_pair(cross_executor):
                                        pass

                                if carry and ready:
                                    ready.appendleft(carry)
                                    carry = None
                                    while _submit_pair(cross_executor):
                                        pass

                                if (
                                    not inflight_meta
                                    and len(ready) > 1
                                    and (len(ready) % 2 == 1)
                                ):
                                    carry = ready.pop()

                        # Finalization
                        if final_collection_path is None:
                            if ready:
                                final_collection_path = ready[0][1]
                            elif carry:
                                final_collection_path = carry[1]

                        final_collection_path = (
                            _normalize_collection_root(final_collection_path)
                            or final_collection_path
                        )
                        if not _is_collection_root(final_collection_path):
                            raise RuntimeError(
                                f"Final path is not a HATS collection root: {final_collection_path}"
                            )

                        _resume_set(
                            resume_log, "phase.crossmatch", "completed", log_cross
                        )
                        log_cross.info(
                            "END crossmatch: parallel tournament done; final root: %s",
                            final_collection_path,
                        )

                except Exception:
                    # Catch ANY driver-side error in the tournament
                    log_cross.exception(
                        "Crossmatch tournament FAILED with an unhandled error"
                    )
                    try:
                        _resume_set(resume_log, "phase.crossmatch", "failed", log_cross)
                    except Exception:
                        pass
                    raise

        else:
            base_logger.error(
                "Unknown combine_mode: %s", combine_mode, extra={"phase": "crossmatch"}
            )
            if _worker_floor_monitor is not None:
                _worker_floor_monitor.stop()
            client.close()
            cluster.close()
            return

        # -----------------------
        # 4) DEDUPLICATION
        # -----------------------
        log_dedup = _phase_logger(base_logger, "deduplication")

        # In "concatenate" mode there is no LSDB collection root; skip dedup entirely.
        if combine_mode == "concatenate":
            log_dedup.info(
                "Skip deduplication: combine_mode='concatenate' (no LSDB collection root)."
            )
        else:
            log_dedup.info(
                "START deduplication: LSDB graph labeling and tie consolidation"
            )

            # Ensure final collection root for dedup
            if not final_collection_path:
                resumed = _resume_get(
                    resume_log, f"crossmatch_step{final_step}.collection_path"
                )
                resumed = _normalize_collection_root(resumed)
                if resumed and _is_collection_root(resumed):
                    final_collection_path = resumed
                    log_dedup.info(
                        "Recovered final collection root from resume: %s",
                        final_collection_path,
                    )
                else:
                    guessed = _guess_collection_for_step(temp_dir, final_step)
                    if guessed:
                        final_collection_path = guessed
                        log_dedup.info(
                            "Guessed final collection root from disk: %s",
                            final_collection_path,
                        )

            use_distributed = final_collection_path is not None and _is_collection_root(
                final_collection_path
            )

            # --- Safe config parsing ---
            tiebreaking_priority_cfg = translation_config.get("tiebreaking_priority")
            if isinstance(tiebreaking_priority_cfg, (str, bytes)):
                tiebreaking_priority_cfg = [str(tiebreaking_priority_cfg)]
            if (
                not isinstance(tiebreaking_priority_cfg, (list, tuple))
                or not tiebreaking_priority_cfg
            ):
                raise TypeError(
                    "tiebreaking_priority must be a non-empty list of column names."
                )

            instrument_type_priority_cfg = translation_config.get(
                "instrument_type_priority"
            )
            if "instrument_type_homogenized" in set(
                tiebreaking_priority_cfg
            ) and not isinstance(instrument_type_priority_cfg, dict):
                raise TypeError(
                    "instrument_type_priority must be a mapping when "
                    "'instrument_type_homogenized' is present in tiebreaking_priority."
                )
            delta_z_threshold_cfg = float(
                translation_config.get("delta_z_threshold", 0.0)
            )

            #######################################################################
            # Diagnostics / outputs
            # - edge_log: enable diagnostics for object types excluded from the graph
            # - group_col: set to None to disable exporting group labels
            edge_log = bool(
                translation_config.get("dedup_edge_diagnostics_enabled", False)
            )
            representative_radius_diagnostics_enabled = bool(
                translation_config.get(
                    "representative_radius_diagnostics_enabled", False
                )
            )
            group_col = "group_id"  # None to deactivate
            #######################################################################

            if use_distributed:
                log_dedup.info(
                    "Running graph-based dedup on final merged collection (Dask merge with catalog ._ddf)"
                )
                final_collection_path = _normalize_collection_root(
                    final_collection_path
                )
                log_dedup.info("final_collection_root: %s", final_collection_path)

                if not _is_collection_root(final_collection_path):
                    raise RuntimeError(
                        f"Expected collection root at: {final_collection_path}"
                    )

                # Open the LSDB collection directly (no subcatalog hunting, no margins here)
                try:
                    final_cat = lsdb.open_catalog(final_collection_path)
                    log_dedup.info("Opened LSDB collection at root path.")
                except Exception as e:
                    raise RuntimeError(
                        f"Could not open LSDB collection at: {final_collection_path}"
                    ) from e

                # Build labels lazily over the catalog
                try:
                    labels_dd = run_dedup_with_lsdb_map_partitions(
                        final_cat,
                        tiebreaking_priority=tiebreaking_priority_cfg,
                        instrument_type_priority=(
                            instrument_type_priority_cfg
                            if isinstance(instrument_type_priority_cfg, dict)
                            else None
                        ),
                        delta_z_threshold=delta_z_threshold_cfg,
                        crd_col="CRD_ID",
                        compared_col="compared_to",
                        z_col="z",
                        tie_col="tie_result",
                        edge_log=edge_log,
                        group_col=group_col,
                        crossmatch_radius_arcsec=crossmatch_radius_arcsec,
                        margin_threshold_arcsec=margin_threshold_arcsec,
                        margin_warning_fraction=margin_warning_fraction,
                        representative_radius_diagnostics_enabled=representative_radius_diagnostics_enabled,
                        object_type_inclusion=object_type_inclusion,
                        max_representative_radius_arcsec=max_representative_radius_arcsec,
                    )
                    log_dedup.info(
                        "Labels graph built (lazy). Persisting compact labels for "
                        "validation and final merge reuse..."
                    )
                    labels_dd = labels_dd.persist()
                    dask_wait(labels_dd)
                    log_dedup.info(
                        "Compact labels persisted: npartitions=%s columns=%d",
                        labels_dd.npartitions,
                        len(labels_dd.columns),
                    )
                except Exception as e:
                    import traceback as _tb

                    log_dedup.error(
                        "FAILED while building labels graph: %s\n%s",
                        repr(e),
                        _tb.format_exc(),
                    )
                    raise

                # Prepare RHS minimal schema (Dask DataFrame)
                try:
                    rhs_dd = labels_dd.rename(columns={"tie_result": "tie_result_new"})
                    keep_cols = ["CRD_ID", "tie_result_new"]
                    if group_col and (group_col in rhs_dd.columns):
                        keep_cols.append(group_col)
                    rhs_dd = rhs_dd[keep_cols]
                    log_dedup.info("Prepared RHS columns: %s", keep_cols)
                except Exception as e:
                    import traceback as _tb

                    log_dedup.error(
                        "FAILED while preparing RHS labels: %s\n%s",
                        repr(e),
                        _tb.format_exc(),
                    )
                    raise

                # Base Dask DataFrame from the LSDB collection
                try:
                    df_all = final_cat._ddf
                    # Align join key dtype defensively
                    try:
                        df_all = df_all.assign(
                            CRD_ID=df_all["CRD_ID"].astype("string[pyarrow]")
                        )
                        rhs_dd = rhs_dd.assign(
                            CRD_ID=rhs_dd["CRD_ID"].astype("string[pyarrow]")
                        )
                    except Exception:
                        df_all = df_all.assign(CRD_ID=df_all["CRD_ID"].astype("string"))
                        rhs_dd = rhs_dd.assign(CRD_ID=rhs_dd["CRD_ID"].astype("string"))
                    log_dedup.info("Aligned CRD_ID dtype on both sides.")
                except Exception as e:
                    import traceback as _tb

                    log_dedup.error(
                        "FAILED while preparing base from catalog: %s\n%s",
                        repr(e),
                        _tb.format_exc(),
                    )
                    raise

                # Clear divisions and perform the distributed merge
                try:
                    try:
                        df_all = df_all.clear_divisions()
                    except Exception:
                        pass
                    try:
                        rhs_dd = rhs_dd.clear_divisions()
                    except Exception:
                        pass
                    with dask.config.set({"dataframe.shuffle.method": "tasks"}):
                        merged = dd.merge(df_all, rhs_dd, on="CRD_ID", how="left")
                    log_dedup.info("Dask merge graph built (lazy).")
                except Exception as e:
                    import traceback as _tb

                    log_dedup.error(
                        "FAILED during Dask merge: %s\n%s", repr(e), _tb.format_exc()
                    )
                    raise

                # Coalesce tie_result in Dask (still lazy)
                try:
                    label_merge_diagnostics = bool(
                        translation_config.get(
                            "label_merge_diagnostics_enabled", True
                        )
                    )
                    missing_new_labels_lazy = (
                        merged["tie_result_new"].isna().sum()
                        if label_merge_diagnostics
                        else None
                    )
                    if "tie_result" in merged.columns:
                        merged["tie_result"] = merged["tie_result_new"].fillna(
                            merged["tie_result"]
                        )
                    else:
                        merged = merged.assign(tie_result=merged["tie_result_new"])
                    if "tie_result_new" in merged.columns:
                        merged = merged.drop(columns=["tie_result_new"])
                    merged["tie_result"] = merged["tie_result"].astype("Int8")

                    validate_edges = bool(
                        translation_config.get("validate_global_graph_edges", False)
                    )
                    validate_ties = bool(
                        translation_config.get("validate_global_tie_invariants", False)
                    )
                    tie_diagnostics = bool(
                        translation_config.get(
                            "tie_invariant_diagnostics_enabled", True
                        )
                    )
                    detailed_tie_diagnostics = bool(
                        translation_config.get(
                            "tie_invariant_diagnostics_detailed_enabled", False
                        )
                    )
                    invalid_stats_lazy = None
                    validation_tasks = []
                    representative_limit_arcsec = (
                        max_representative_radius_arcsec
                        if max_representative_radius_arcsec is not None
                        else crossmatch_radius_arcsec
                    )
                    if representative_radius_diagnostics_enabled:
                        representative_radius = labels_dd[
                            REPRESENTATIVE_RADIUS_DIAGNOSTIC_COLUMN
                        ]
                        representative_radius_valid = representative_radius.dropna()
                        validation_tasks.extend(
                            [
                                representative_radius_valid.count(),
                                representative_radius_valid.gt(
                                    representative_limit_arcsec
                                ).sum(),
                                representative_radius_valid.gt(
                                    2.0 * representative_limit_arcsec
                                ).sum(),
                                representative_radius_valid.gt(
                                    margin_threshold_arcsec
                                ).sum(),
                                representative_radius_valid.max(),
                            ]
                        )
                    if validate_edges:
                        mismatch_lazy, dangling_lazy = (
                            count_global_edge_group_mismatches(merged)
                        )
                        validation_tasks.extend([mismatch_lazy, dangling_lazy])
                    if label_merge_diagnostics:
                        validation_tasks.append(missing_new_labels_lazy)
                    if validate_ties:
                        if tie_diagnostics or detailed_tie_diagnostics:
                            invalid_stats_lazy, missing_group_rows_lazy = (
                                build_global_tie_invariant_diagnostics(labels_dd)
                            )
                            validation_tasks.extend(
                                [
                                    invalid_stats_lazy.map_partitions(len).sum()
                                    + missing_group_rows_lazy,
                                    missing_group_rows_lazy,
                                    invalid_stats_lazy["multiple_winners"].sum(),
                                    invalid_stats_lazy["no_survivor"].sum(),
                                    invalid_stats_lazy[
                                        "mixed_winner_hard_tie"
                                    ].sum(),
                                    invalid_stats_lazy["single_hard_tie"].sum(),
                                    invalid_stats_lazy["invalid_tie_values"].sum(),
                                ]
                            )
                        else:
                            validation_tasks.append(
                                count_global_tie_invariant_violations(labels_dd)
                            )

                    validation_results = iter(dask.compute(*validation_tasks))
                    if representative_radius_diagnostics_enabled:
                        representative_components = int(next(validation_results))
                        representative_exceed_radius = int(next(validation_results))
                        representative_exceed_twice_radius = int(
                            next(validation_results)
                        )
                        representative_exceed_margin = int(next(validation_results))
                        representative_max_radius = float(next(validation_results))
                        representative_fraction = (
                            representative_exceed_radius / representative_components
                            if representative_components
                            else 0.0
                        )
                        log_dedup.info(
                            "Representative-radius diagnostics: components=%d "
                            "radius=%.3farcsec exceeding_radius=%d "
                            "fraction_exceeding=%.6f exceeding_twice_radius=%d "
                            "exceeding_margin=%d max_radius=%.4farcsec",
                            representative_components,
                            representative_limit_arcsec,
                            representative_exceed_radius,
                            representative_fraction,
                            representative_exceed_twice_radius,
                            representative_exceed_margin,
                            representative_max_radius,
                        )
                    if validate_edges:
                        mismatch_count = int(next(validation_results))
                        dangling_count = int(next(validation_results))
                        log_dedup.info(
                            "Global graph validation: cross_group_edges=%d "
                            "dangling_participating_edges=%d",
                            mismatch_count,
                            dangling_count,
                        )
                        if mismatch_count:
                            if max_representative_radius_arcsec is None:
                                raise RuntimeError(
                                    "Partition-local deduplication produced "
                                    f"{mismatch_count} participating edges whose "
                                    "endpoints have different canonical group_id "
                                    "values. Increase margin_threshold_arcsec or "
                                    "inspect long components."
                                )
                            log_dedup.info(
                                "Cross-group edges are expected with representative-"
                                "radius truncation enabled (radius=%.3f arcsec).",
                                max_representative_radius_arcsec,
                            )
                        if dangling_count:
                            raise RuntimeError(
                                "The deduplication graph contains "
                                f"{dangling_count} edge(s) referencing CRD_ID values "
                                "that are absent from the consolidated catalog. "
                                "Inspect compared_to propagation, filtering, and "
                                "catalog concatenation."
                            )

                    if label_merge_diagnostics:
                        missing_new_labels = int(next(validation_results))
                        log_dedup.info(
                            "Label merge diagnostics: rows_without_new_label=%d",
                            missing_new_labels,
                        )

                    if validate_ties:
                        invalid_groups = int(next(validation_results))
                        if tie_diagnostics or detailed_tie_diagnostics:
                            missing_group_rows = int(next(validation_results))
                            multiple_winners = int(next(validation_results))
                            no_survivor = int(next(validation_results))
                            mixed_winner_hard_tie = int(next(validation_results))
                            single_hard_tie = int(next(validation_results))
                            invalid_tie_values = int(next(validation_results))
                            log_dedup.info(
                                "Global tie invariant diagnostics: "
                                "missing_group_rows=%d multiple_winners=%d "
                                "no_survivor=%d mixed_winner_hard_tie=%d "
                                "single_hard_tie=%d invalid_tie_values=%d",
                                missing_group_rows,
                                multiple_winners,
                                no_survivor,
                                mixed_winner_hard_tie,
                                single_hard_tie,
                                invalid_tie_values,
                            )
                        log_dedup.info(
                            "Global tie invariant validation: invalid_groups=%d",
                            invalid_groups,
                        )
                        if (
                            invalid_groups
                            and detailed_tie_diagnostics
                            and invalid_stats_lazy is not None
                        ):
                            sample_size = int(
                                translation_config.get(
                                    "tie_invariant_diagnostics_sample_size", 10
                                )
                            )
                            max_rows = int(
                                translation_config.get(
                                    "tie_invariant_diagnostics_max_rows", 100
                                )
                            )
                            invalid_sample = invalid_stats_lazy.reset_index().head(
                                sample_size, npartitions=-1
                            )
                            log_dedup.error(
                                "Invalid tie group summary sample: %s",
                                invalid_sample.to_dict("records"),
                            )
                            sample_group_ids = invalid_sample["group_id"].tolist()
                            if sample_group_ids:
                                member_columns = [
                                    column
                                    for column in (
                                        "group_id",
                                        "CRD_ID",
                                        "tie_result",
                                        "z_flag_homogenized",
                                        "object_type_homogenized",
                                    )
                                    if column in labels_dd.columns
                                ]
                                member_sample = labels_dd.loc[
                                    labels_dd["group_id"].isin(sample_group_ids),
                                    member_columns,
                                ].head(max_rows, npartitions=-1)
                                log_dedup.error(
                                    "Invalid tie group member sample: %s",
                                    member_sample.to_dict("records"),
                                )
                        if invalid_groups:
                            raise RuntimeError(
                                f"Global tie-result invariants failed for "
                                f"{invalid_groups} group(s)"
                            )
                    log_dedup.info("Coalesced tie_result (lazy).")
                except Exception as e:
                    import traceback as _tb

                    log_dedup.error(
                        "FAILED while coalescing tie_result: %s\n%s",
                        repr(e),
                        _tb.format_exc(),
                    )
                    raise

                # Final materialization: avoid pulling the full dataframe to the
                # driver when HATS output can consume a Dask dataframe directly.
                try:
                    if output_format in {"hats", "parquet", "csv"}:
                        df_final = merged
                        log_dedup.info(
                            "Keeping final merged dataframe lazy for distributed "
                            "output format=%s (npartitions=%s).",
                            output_format,
                            getattr(df_final, "npartitions", "unknown"),
                        )
                    else:
                        df_final = merged.compute()

                        if "group_id" in df_final.columns:
                            dup = df_final["group_id"].value_counts(dropna=True)
                            n_big = int((dup > 1).sum())
                            log_dedup.info(
                                "Sanity group_id: uniques=%d, ids_with_>1_occurrences=%d",
                                int(dup.size),
                                n_big,
                            )
                        else:
                            log_dedup.info(
                                "No group_id column present after dedup; skipping sanity counts."
                            )

                    current_workers = len(client.scheduler_info().get("workers", {}))
                    log_dedup.info("WORKERS STILL RUNNING=%d.", current_workers)
                    log_dedup.info(
                        "END deduplication: labels merged back into final dataframe (Dask)"
                    )
                except Exception as e:
                    import traceback as _tb

                    log_dedup.error(
                        "FAILED while computing final dataframe: %s\n%s",
                        repr(e),
                        _tb.format_exc(),
                    )
                    raise

            else:
                # We require a valid LSDB collection root; Parquet fallback is no longer supported.
                raise RuntimeError(
                    "Cannot run dedup: missing or invalid LSDB collection root; Parquet fallback removed."
                )

    # -----------------------
    # 5) CONSOLIDATION / EXPORT
    # -----------------------
    log_cons = _phase_logger(base_logger, "consolidation")
    log_cons.info(
        "START consolidation: staging artifacts into process dir (base_dir=%s)",
        base_dir,
    )
    lazy_distributed_output = output_format in {
        "hats",
        "parquet",
        "csv",
    } and _is_dask_dataframe(df_final)

    log_cons.info(
        "Final dataframe backend=%s npartitions=%s columns=%d",
        "dask" if _is_dask_dataframe(df_final) else "pandas",
        getattr(df_final, "npartitions", 1),
        len(df_final.columns),
    )

    if combine_mode == "concatenate" and "tie_result" in df_final.columns:
        log_cons.info("Dropping 'tie_result' (concatenate mode)")
        try:
            df_final = df_final.drop(columns=["tie_result"])
        except Exception as e:
            log_cons.error("FAILED dropping 'tie_result' in concatenate mode: %s", e)
            raise

    if combine_mode == "concatenate_and_remove_duplicates":
        tie_treatment_option = (
            str(param_config.get("tie_treatment_option", "remove_all") or "remove_all")
            .strip()
            .lower()
        )
        if tie_treatment_option not in {"remove_all", "keep_all", "draw_one"}:
            log_cons.warning(
                "Unknown tie_treatment_option=%s; defaulting to remove_all.",
                tie_treatment_option,
            )
            tie_treatment_option = "remove_all"

        if lazy_distributed_output:
            if "tie_result" not in df_final.columns:
                raise RuntimeError(
                    "Expected 'tie_result' column for remove-duplicates mode"
                )
            else:
                log_cons.info(
                    "Applying lazy Dask tie_result filter for distributed output "
                    "(tie_treatment_option=%s).",
                    tie_treatment_option,
                )
                try:
                    requested_option = tie_treatment_option
                    df_final, tie_treatment_option = filter_dask_by_tie_treatment(
                        df_final,
                        tie_treatment_option,
                    )
                    if (
                        requested_option == "draw_one"
                        and tie_treatment_option != "draw_one"
                    ):
                        log_cons.info(
                            "draw_one requires CRD_ID and group_id; falling back "
                            "to remove_all."
                        )
                    elif tie_treatment_option == "draw_one":
                        log_cons.info(
                            "Resolving hard ties lazily and deterministically by "
                            "the smallest CRD_ID in each group."
                        )
                    elif tie_treatment_option == "keep_all":
                        log_cons.info("Filtering lazily by tie_result in {1,2}.")
                    else:
                        log_cons.info("Filtering lazily by tie_result == 1.")
                except Exception as e:
                    log_cons.error(
                        "FAILED while applying lazy tie_result filter: %s", e
                    )
                    raise

        if lazy_distributed_output:
            pass
        elif "tie_result" not in df_final.columns:
            raise RuntimeError(
                "Expected 'tie_result' column for remove-duplicates mode"
            )
        else:
            log_cons.info(
                "tie_treatment_option=%s (remove-duplicates mode)",
                tie_treatment_option,
            )
            try:
                requested_option = tie_treatment_option
                df_final, tie_treatment_option, n_groups = (
                    filter_pandas_by_tie_treatment(
                        df_final,
                        tie_treatment_option,
                    )
                )
                if (
                    requested_option == "draw_one"
                    and tie_treatment_option != "draw_one"
                ):
                    log_cons.warning(
                        "tie_treatment_option=draw_one requires group_id; "
                        "falling back to remove_all."
                    )
                elif tie_treatment_option == "draw_one":
                    log_cons.info("Draw-one resolved %d absolute-tie groups.", n_groups)
            except Exception as e:
                log_cons.error("FAILED while filtering by tie_treatment_option: %s", e)
                raise

    if not lazy_distributed_output:
        _ensure_non_empty_final_dataframe(
            df_final,
            log_cons,
            context="immediately before final export",
        )
    else:
        log_cons.info(
            "Deferring empty-output validation until distributed parquet staging."
        )

    # Stage final output with your save_dataframe
    staged_output_base = os.path.join(base_dir, output_name)
    log_cons.info(
        "About to call save_dataframe(base=%s, format=%s)",
        staged_output_base,
        output_format,
    )
    try:
        save_dataframe(
            df_final,
            staged_output_base,
            output_format,
            temp_dir=temp_dir,
            client=client,
            logger=log_cons,
        )
        log_cons.info("Staged final output at %s.%s", staged_output_base, output_format)
    except Exception as e:
        import traceback as _tb

        log_cons.error("FAILED in save_dataframe: %s\n%s", repr(e), _tb.format_exc())
        raise

    relative_path = os.path.join(output_dir, f"{output_name}.{output_format}")

    expected_columns = ["id", "ra", "dec", "z", "z_flag", "z_err", "survey"]
    columns_assoc = {}

    # Special handling for id
    try:
        if "CRD_ID" in df_final.columns:
            columns_assoc["id"] = "CRD_ID"
        elif "id" in df_final.columns:
            columns_assoc["id"] = "id"
    except Exception as e:
        log_cons.debug("While mapping 'id' column: %s", e)

    # Special handling for z_flag
    try:
        if "z_flag_homogenized" in df_final.columns:
            columns_assoc["z_flag"] = "z_flag_homogenized"
        elif "z_flag" in df_final.columns:
            columns_assoc["z_flag"] = "z_flag"
    except Exception as e:
        log_cons.debug("While mapping 'z_flag' column: %s", e)

    # Identity mapping for the others
    try:
        for col in expected_columns:
            if col not in ("id", "z_flag") and col in df_final.columns:
                columns_assoc[col] = col
        log_cons.info("columns_assoc: %s", columns_assoc)
    except Exception as e:
        log_cons.debug("While building columns_assoc: %s", e)

    # Update process info
    try:
        if _worker_floor_monitor is not None and _worker_floor_monitor.timed_out:
            raise RuntimeError(
                "Dask worker count stayed below the configured minimum beyond "
                "the recovery timeout."
            )
        update_process_info(
            process_info,
            process_info_path,
            "outputs",
            [
                {
                    "path": relative_path,
                    "root_dir": output_root_dir,
                    "role": "main",
                    "columns_assoc": columns_assoc,
                }
            ],
        )
        update_process_info(
            process_info, process_info_path, "end_time", str(pd.Timestamp.now())
        )
        update_process_info(process_info, process_info_path, "status", "Successful")
        log_cons.info("Process info updated with output: %s", relative_path)
    except Exception as e:
        import traceback as _tb

        log_cons.error("FAILED to update process_info: %s\n%s", e, _tb.format_exc())
        raise

    # Record resource peaks before process_info is copied to the published output.
    _resource_usage_monitor.report(_phase_logger(base_logger, "resources"))

    # -----------------------
    # Publish step
    # -----------------------
    publish_logger = _phase_logger(base_logger, "register")
    publish_logger.info(
        "START publish: copying staged artifacts from process dir to out_root_and_dir (%s)",
        out_root_and_dir,
    )

    try:
        _snapshot_shell_logs(
            str(launch_dir), launch_snap_dir, publish_logger, max_size_mb=100
        )
        publish_logger.info(
            "Snapshotted shell logs from %s into %s", launch_dir, launch_snap_dir
        )
    except Exception as e:
        publish_logger.warning("Snapshot of shell logs failed: %s", e)

    # Create publish dir only now
    try:
        os.makedirs(out_root_and_dir, exist_ok=True)
        publish_logger.info("Ensured publish dir exists: %s", out_root_and_dir)
    except Exception as e:
        publish_logger.error(
            "FAILED to create publish dir '%s': %s", out_root_and_dir, e
        )
        raise

    src_out = f"{staged_output_base}.{output_format}"
    dst_out = os.path.join(out_root_and_dir, f"{output_name}.{output_format}")
    publish_artifacts = [
        (
            "tree",
            os.path.join(base_dir, "process_info"),
            os.path.join(out_root_and_dir, "process_info"),
        ),
        (
            "file",
            os.path.join(base_dir, "process.yml"),
            os.path.join(out_root_and_dir, "process.yml"),
        ),
        (
            "tree" if output_format == "hats" else "file",
            src_out,
            dst_out,
        ),
    ]
    if os.path.exists(os.path.join(base_dir, "process.yaml")):
        publish_artifacts.append(
            (
                "file",
                os.path.join(base_dir, "process.yaml"),
                os.path.join(out_root_and_dir, "process.yaml"),
            )
        )
    if os.path.exists(os.path.join(base_dir, "config.yaml")):
        publish_artifacts.append(
            (
                "file",
                os.path.join(base_dir, "config.yaml"),
                os.path.join(out_root_and_dir, "config.yaml"),
            )
        )
    if os.path.exists(path_to_translation_file):
        publish_artifacts.append(
            (
                "file",
                path_to_translation_file,
                os.path.join(out_root_and_dir, "flags_translation.yaml"),
            )
        )

    _copy_and_verify_publish_artifacts(publish_artifacts, publish_logger)

    publish_logger.info(
        "END publish: artifacts copied and verified at %s", out_root_and_dir
    )

    if delete_temp_files:
        try:
            shutil.rmtree(temp_dir)
            log_cons.info(
                "Deleted entire temp_dir after successful pipeline completion: %s",
                temp_dir,
            )
        except Exception as e:
            log_cons.warning("Could not delete temp_dir %s: %s", temp_dir, e)

    log_cons.info("END consolidation: export complete")
    if _worker_floor_monitor is not None:
        _worker_floor_monitor.stop()
    client.close()
    cluster.close()


# -----------------------
# Helpers
# -----------------------
def _pick_next_process_dir(root: str) -> str:
    """Return first non-existing 'processNNN' path under root."""
    i = 1
    while True:
        name = f"process{i:03d}"
        path = os.path.join(root, name)
        if not os.path.exists(path):
            return path
        i += 1


# -----------------------
# CLI
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine redshift catalogs via preparation, crossmatch (no tie-breaking), and graph-based deduplication."
    )
    parser.add_argument("config_path", help="Path to YAML config file.")
    parser.add_argument(
        "--cwd", default=os.getcwd(), help="Working directory (default: current dir)."
    )
    parser.add_argument(
        "--base_dir",
        default=None,
        help="Base directory for outputs and logs. If omitted, picks processNNN under --cwd.",
    )
    args = parser.parse_args()

    # Resolve working dir and base_dir
    workdir = os.path.abspath(args.cwd)
    os.makedirs(workdir, exist_ok=True)

    base_dir = args.base_dir
    if not base_dir:
        base_dir = _pick_next_process_dir(workdir)

    base_dir = os.path.abspath(base_dir)
    os.makedirs(base_dir, exist_ok=True)

    # Optional: echo the chosen run dir early for visibility when logs aren't wired yet
    print(f"Using run directory: {base_dir}")

    start_ts = time.time()
    ok = False
    try:
        # Pass resolved paths
        main(args.config_path, workdir, base_dir)
        ok = True
    finally:
        dur = time.time() - start_ts
        lg = logging.getLogger("crc")
        if _resource_usage_monitor is not None:
            _resource_usage_monitor.report(
                logging.LoggerAdapter(lg, {"phase": "resources"})
            )
        if _worker_floor_monitor is not None:
            _worker_floor_monitor.stop()
        msg = f"Pipeline {'completed successfully' if ok else 'terminated with errors'} in {dur:.2f} seconds. (run dir: {base_dir})"
        if lg.handlers:
            logging.LoggerAdapter(lg, {"phase": "consolidation"}).info(msg)
        else:
            print(("OK " if ok else "FAIL ") + msg)
