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


def test_slurm_executor_uses_fixed_allocation_by_default(monkeypatch):
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
    assert cluster.adapt_calls == []


def test_slurm_executor_enables_adaptive_scaling_only_explicitly(monkeypatch):
    monkeypatch.setattr(executor, "SLURMCluster", FakeSlurmCluster)

    cluster = executor.get_executor(
        {
            "name": "slurm",
            "args": {
                "instance": {"cores": 2, "processes": 1, "memory": "30GB"},
                "scale": {
                    "minimum_jobs": 7,
                    "maximum_jobs": 15,
                    "adaptive": True,
                },
            },
        }
    )

    assert cluster.adapt_calls == [{"minimum_jobs": 7, "maximum_jobs": 15}]
