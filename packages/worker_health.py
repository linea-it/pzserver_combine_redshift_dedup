"""Lightweight monitoring for the minimum Dask worker floor."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any


class WorkerFloorMonitor:
    """Abort a run when the cluster stays below its configured worker floor."""

    def __init__(
        self,
        client: Any,
        minimum_workers: int,
        recovery_timeout_seconds: float = 600.0,
        check_interval_seconds: float = 10.0,
        logger: logging.LoggerAdapter | None = None,
        on_timeout: Callable[[], None] | None = None,
    ) -> None:
        self.client = client
        self.minimum_workers = minimum_workers
        self.recovery_timeout_seconds = recovery_timeout_seconds
        self.check_interval_seconds = check_interval_seconds
        self.logger = logger or logging.LoggerAdapter(
            logging.getLogger("crc.worker_health"), {"phase": "resources"}
        )
        self.on_timeout = on_timeout
        self._below_since: float | None = None
        self._timed_out = False
        self._query_failure_logged = False
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def timed_out(self) -> bool:
        """Whether the worker floor failed to recover within the timeout."""
        return self._timed_out

    def check_once(self, now: float | None = None) -> None:
        """Evaluate one scheduler snapshot."""
        if self._timed_out or self.minimum_workers <= 0:
            return
        now = time.monotonic() if now is None else now
        try:
            current_workers = len(self.client.scheduler_info().get("workers", {}))
        except Exception as exc:
            if not self._query_failure_logged:
                self.logger.warning(
                    "Could not query worker count; worker-floor timer unchanged: %s",
                    exc,
                )
                self._query_failure_logged = True
            return

        self._query_failure_logged = False
        if current_workers >= self.minimum_workers:
            if self._below_since is not None:
                self.logger.info(
                    "Worker floor recovered: current=%d minimum=%d after %.1fs.",
                    current_workers,
                    self.minimum_workers,
                    now - self._below_since,
                )
                self._below_since = None
            return

        if self._below_since is None:
            self._below_since = now
            self.logger.warning(
                "Worker count below configured floor: current=%d minimum=%d; "
                "waiting up to %.1fs for adaptive recovery.",
                current_workers,
                self.minimum_workers,
                self.recovery_timeout_seconds,
            )
            return

        elapsed = now - self._below_since
        if elapsed < self.recovery_timeout_seconds:
            return

        self._timed_out = True
        self.logger.error(
            "Worker floor did not recover: current=%d minimum=%d elapsed=%.1fs; "
            "aborting pipeline.",
            current_workers,
            self.minimum_workers,
            elapsed,
        )
        if self.on_timeout is not None:
            try:
                self.on_timeout()
            except Exception:
                self.logger.exception("Failed to close the degraded Dask cluster.")

    def start(self) -> None:
        """Start monitoring in a daemon thread."""
        if self._thread is not None or self.minimum_workers <= 0:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="crc-worker-floor-monitor",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stop_event.is_set() and not self._timed_out:
            self.check_once()
            self._stop_event.wait(self.check_interval_seconds)

    def stop(self) -> None:
        """Stop monitoring without triggering the recovery callback."""
        self._stop_event.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=max(1.0, self.check_interval_seconds * 2))
