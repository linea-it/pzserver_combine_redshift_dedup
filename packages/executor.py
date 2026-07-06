"""Executor helpers for CRC.

Provides a small factory to create Dask clusters from config.

Public API:
    - get_executor
"""

# -----------------------
# Standard library
# -----------------------
import logging
import math
from typing import Any, Dict

# -----------------------
# Third-party
# -----------------------
from dask.distributed import LocalCluster
from dask_jobqueue import SLURMCluster

# -----------------------
# Project
# -----------------------
from utils import get_phase_logger

__all__ = ["get_executor"]

LOGGER_NAME = "crc.executor"


def _get_logger() -> logging.LoggerAdapter:
    """Return a phase-aware logger ('crc.executor' with phase='executor').

    Returns:
        logging.LoggerAdapter: Logger with phase context.
    """
    base = logging.getLogger(LOGGER_NAME)
    base.propagate = True
    return get_phase_logger("executor", base)


def get_executor(executor_config: Dict[str, Any], logs_dir: str | None = None):
    """Create and return a Dask cluster from config.

    Behavior:
      - Local: start a LocalCluster(**args).
      - SLURM: start SLURMCluster with n_workers = minimum_jobs * processes.
        Each job will use the provided cores, processes, and memory.
        Always enable adaptive scaling after submitting the minimum jobs.

    Args:
        executor_config: Dictionary with:
            - name: "local" or "slurm".
            - args:
                For "local": passed to LocalCluster(**args).
                For "slurm":
                  - instance: kwargs for SLURMCluster(**instance).
                  - scale:
                      * minimum_jobs (int)
                      * maximum_jobs (int)
                      * adaptive_interval_seconds (float)
                      * adaptive_scale_down_delay_seconds (float)
        logs_dir: Directory to store SLURM job logs (if provided).

    Returns:
        LocalCluster | SLURMCluster: Configured Dask cluster.
    """
    logger = _get_logger()

    executor_name = executor_config.get("name", "local")
    args = executor_config.get("args", {}) or {}

    logger.info("Setting up executor: %s", executor_name)

    # -----------------------
    # Local cluster
    # -----------------------
    if executor_name == "local":
        cluster = LocalCluster(**args)
        logger.info("LocalCluster started with args=%s", args)
        return cluster

    # -----------------------
    # SLURM cluster
    # -----------------------
    if executor_name == "slurm":
        instance_cfg = dict(args.get("instance", {}) or {})
        scale_cfg = dict(args.get("scale", {}) or {})

        # Add job log directives if requested
        if logs_dir:
            extra_directives = list(instance_cfg.get("job_extra_directives", []))
            extra_directives.extend(
                [
                    f"--output={logs_dir}/slurm-%j.out",
                    f"--error={logs_dir}/slurm-%j.err",
                ]
            )
            instance_cfg["job_extra_directives"] = extra_directives

        processes = int(instance_cfg.get("processes", 1))
        cores = instance_cfg.get("cores")
        memory = instance_cfg.get("memory")
        queue = instance_cfg.get("queue")
        account = instance_cfg.get("account")

        min_jobs = int(scale_cfg.get("minimum_jobs", 0))
        max_jobs = int(scale_cfg.get("maximum_jobs", 0) or 0)
        adaptive_interval_seconds = float(
            scale_cfg.get("adaptive_interval_seconds", 10.0)
        )
        adaptive_scale_down_delay_seconds = float(
            scale_cfg.get("adaptive_scale_down_delay_seconds", 180.0)
        )
        # Keep direct executor callers using the former low-level setting
        # working while exposing the clearer duration-based configuration.
        if "adaptive_scale_down_delay_seconds" not in scale_cfg and (
            "adaptive_wait_count" in scale_cfg
        ):
            adaptive_wait_count = int(scale_cfg["adaptive_wait_count"])
            adaptive_scale_down_delay_seconds = (
                adaptive_wait_count * adaptive_interval_seconds
            )
        else:
            adaptive_wait_count = max(
                1,
                math.ceil(
                    adaptive_scale_down_delay_seconds / adaptive_interval_seconds
                ),
            )
        # n_workers in SLURMCluster = total worker processes
        n_workers_init = min_jobs * processes

        logger.info(
            "SLURM job template: cores=%s, processes=%s, memory=%s, queue=%s, account=%s",
            cores,
            processes,
            memory,
            queue,
            account,
        )
        logger.info(
            "Initial submit: minimum_jobs=%d -> n_workers=%d (worker processes).",
            min_jobs,
            n_workers_init,
        )

        # Submit jobs immediately on construction
        cluster = SLURMCluster(n_workers=n_workers_init, **instance_cfg)
        logger.info("SLURMCluster started with instance args=%s", instance_cfg)

        # Keep adaptive scaling enabled even when min == max. In that case it
        # acts as a fixed-size allocation that can replace jobs lost mid-run.
        cluster.adapt(
            minimum_jobs=min_jobs,
            maximum_jobs=max_jobs,
            interval=f"{adaptive_interval_seconds:g}s",
            wait_count=adaptive_wait_count,
        )
        logger.info(
            "Adaptive scaling enabled: minimum_jobs=%d maximum_jobs=%d "
            "interval=%.1fs scale_down_delay=%.1fs wait_count=%d "
            "effective_scale_down_delay=%.1fs",
            min_jobs,
            max_jobs,
            adaptive_interval_seconds,
            adaptive_scale_down_delay_seconds,
            adaptive_wait_count,
            adaptive_wait_count * adaptive_interval_seconds,
        )

        return cluster

    # -----------------------
    # Unknown -> minimal local
    # -----------------------
    logger.warning(
        "Unknown executor '%s'. Falling back to a minimal LocalCluster.", executor_name
    )
    return LocalCluster(n_workers=1, threads_per_worker=1)
