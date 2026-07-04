import logging

from worker_health import WorkerFloorMonitor


class SnapshotClient:
    def __init__(self, worker_counts):
        self.worker_counts = iter(worker_counts)

    def scheduler_info(self):
        count = next(self.worker_counts)
        return {"workers": {f"worker-{index}": {} for index in range(count)}}


def _monitor(worker_counts, on_timeout=None):
    return WorkerFloorMonitor(
        SnapshotClient(worker_counts),
        minimum_workers=3,
        recovery_timeout_seconds=600,
        logger=logging.LoggerAdapter(logging.getLogger("test"), {}),
        on_timeout=on_timeout,
    )


def test_worker_floor_timer_resets_after_recovery():
    monitor = _monitor([2, 3, 2])

    monitor.check_once(now=0)
    monitor.check_once(now=100)
    monitor.check_once(now=200)

    assert monitor._below_since == 200
    assert not monitor.timed_out


def test_worker_floor_does_not_abort_before_timeout():
    called = []
    monitor = _monitor([2, 2], on_timeout=lambda: called.append(True))

    monitor.check_once(now=0)
    monitor.check_once(now=599)

    assert not monitor.timed_out
    assert called == []


def test_worker_floor_aborts_after_timeout():
    called = []
    monitor = _monitor([2, 2], on_timeout=lambda: called.append(True))

    monitor.check_once(now=0)
    monitor.check_once(now=600)

    assert monitor.timed_out
    assert called == [True]
