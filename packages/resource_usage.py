"""Process memory peak monitoring for the CRC pipeline."""

from __future__ import annotations

import logging
import resource
import sys
import threading
from typing import Any


def _format_bytes(value: int) -> str:
    """Format a byte count using binary units."""
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024.0 or unit == "TiB":
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TiB"


def _driver_peak_rss_bytes() -> int:
    """Return the OS-recorded peak RSS for the current (driver) process."""
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Linux reports KiB; macOS reports bytes.
    return peak if sys.platform == "darwin" else peak * 1024


class ResourceUsageMonitor:
    """Track driver and per-worker RSS peaks over a pipeline run.

    Dask exposes each worker process RSS in scheduler heartbeats. Sampling the
    scheduler also retains peaks for workers later removed by adaptive scaling.
    """

    def __init__(self, client: Any, interval_seconds: float = 1.0) -> None:
        self.client = client
        self.interval_seconds = interval_seconds
        self.worker_peaks: dict[str, int] = {}
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._reported = False

    def sample_once(self) -> None:
        """Capture one scheduler snapshot without discarding prior peaks."""
        workers = self.client.scheduler_info().get("workers", {})
        for address, info in workers.items():
            memory = (info.get("metrics") or {}).get("memory")
            if memory is None:
                continue
            memory = int(memory)
            self.worker_peaks[address] = max(memory, self.worker_peaks.get(address, 0))

    def start(self) -> None:
        """Start background sampling."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="crc-resource-usage-monitor",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.sample_once()
            except Exception:
                # A transient scheduler query failure must not affect the pipeline.
                pass
            self._stop_event.wait(self.interval_seconds)

    def report(self, logger: logging.LoggerAdapter) -> None:
        """Stop sampling and write one overall peak-memory summary."""
        if self._reported:
            return
        try:
            self.sample_once()
        except Exception:
            pass
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_seconds * 2))

        driver_peak = _driver_peak_rss_bytes()
        if self.worker_peaks:
            worker, worker_peak = max(
                self.worker_peaks.items(), key=lambda item: item[1]
            )
            logger.info(
                "MEMORY PEAKS: driver_rss=%s; max_worker_rss=%s; "
                "max_worker=%s; workers_observed=%d; worker_sample_interval=%.1fs",
                _format_bytes(driver_peak),
                _format_bytes(worker_peak),
                worker,
                len(self.worker_peaks),
                self.interval_seconds,
            )
        else:
            logger.info(
                "MEMORY PEAKS: driver_rss=%s; max_worker_rss=unavailable; "
                "workers_observed=0; worker_sample_interval=%.1fs",
                _format_bytes(driver_peak),
                self.interval_seconds,
            )
        self._reported = True
