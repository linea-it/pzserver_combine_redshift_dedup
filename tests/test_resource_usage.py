from unittest.mock import Mock, patch

from resource_usage import ResourceUsageMonitor, _format_bytes


class SnapshotClient:
    def __init__(self, snapshots):
        self.snapshots = iter(snapshots)
        self.last = {"workers": {}}

    def scheduler_info(self):
        try:
            self.last = next(self.snapshots)
        except StopIteration:
            pass
        return self.last


def _snapshot(**workers):
    return {
        "workers": {
            address: {"metrics": {"memory": memory}}
            for address, memory in workers.items()
        }
    }


def test_monitor_retains_peak_for_removed_workers():
    client = SnapshotClient(
        [
            _snapshot(worker_a=100, worker_b=250),
            _snapshot(worker_a=300),
            _snapshot(worker_a=200),
        ]
    )
    monitor = ResourceUsageMonitor(client)

    monitor.sample_once()
    monitor.sample_once()
    monitor.sample_once()

    assert monitor.worker_peaks == {"worker_a": 300, "worker_b": 250}


def test_report_logs_driver_and_largest_worker_peak():
    client = SnapshotClient([_snapshot(worker_a=2 * 1024**3)])
    monitor = ResourceUsageMonitor(client)
    logger = Mock()

    with patch("resource_usage._driver_peak_rss_bytes", return_value=3 * 1024**3):
        monitor.report(logger)

    message, *values = logger.info.call_args.args
    assert message.startswith("MEMORY PEAKS:")
    assert values[:3] == ["3.00 GiB", "2.00 GiB", "worker_a"]


def test_format_bytes_uses_binary_units():
    assert _format_bytes(1536 * 1024) == "1.50 MiB"
