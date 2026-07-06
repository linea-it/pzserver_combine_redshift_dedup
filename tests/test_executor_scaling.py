import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages"))

import executor  # noqa: E402


class FakeSlurmCluster:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.adapt_calls = []

    def adapt(self, **kwargs):
        self.adapt_calls.append(kwargs)


def test_slurm_executor_starts_minimum_workers_and_always_enables_adapt(monkeypatch):
    monkeypatch.setattr(executor, "SLURMCluster", FakeSlurmCluster)

    cluster = executor.get_executor(
        {
            "name": "slurm",
            "args": {
                "instance": {"cores": 2, "processes": 1, "memory": "40GB"},
                "scale": {"minimum_jobs": 11, "maximum_jobs": 11},
            },
        }
    )

    assert cluster.kwargs["n_workers"] == 11
    assert cluster.adapt_calls == [
        {
            "minimum_jobs": 11,
            "maximum_jobs": 11,
            "interval": "10s",
            "wait_count": 18,
        }
    ]


def test_slurm_executor_supports_different_adaptive_limits(monkeypatch):
    monkeypatch.setattr(executor, "SLURMCluster", FakeSlurmCluster)

    cluster = executor.get_executor(
        {
            "name": "slurm",
            "args": {
                "instance": {"cores": 2, "processes": 1, "memory": "30GB"},
                "scale": {
                    "minimum_jobs": 7,
                    "maximum_jobs": 15,
                    "adaptive_interval_seconds": 5,
                    "adaptive_scale_down_delay_seconds": 121,
                },
            },
        }
    )

    assert cluster.adapt_calls == [
        {
            "minimum_jobs": 7,
            "maximum_jobs": 15,
            "interval": "5s",
            "wait_count": 25,
        }
    ]


def test_slurm_executor_supports_legacy_adaptive_wait_count(monkeypatch):
    monkeypatch.setattr(executor, "SLURMCluster", FakeSlurmCluster)

    cluster = executor.get_executor(
        {
            "name": "slurm",
            "args": {
                "instance": {"cores": 2, "processes": 1, "memory": "20GB"},
                "scale": {
                    "minimum_jobs": 3,
                    "maximum_jobs": 22,
                    "adaptive_interval_seconds": 10,
                    "adaptive_wait_count": 18,
                },
            },
        }
    )

    assert cluster.adapt_calls == [
        {
            "minimum_jobs": 3,
            "maximum_jobs": 22,
            "interval": "10s",
            "wait_count": 18,
        }
    ]
