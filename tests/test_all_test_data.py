from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pandas as pd
import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "test_configs" / "config.all.yaml"
RUN_SCRIPT = REPO_ROOT / "run.sh"
ENV_NAME = "pipe_crd"

EXPECTED_STATS = {
    "tie_result": {
        0: 1451,
        1: 16688,
        2: 208,
        3: 351,
    },
    "z_flag_homogenized": {
        0.0: 1646,
        1.0: 1793,
        2.0: 1088,
        3.0: 6612,
        4.0: 7208,
        6.0: 351,
    },
    "instrument_type_homogenized": {
        "g": 1152,
        "p": 3838,
        "s": 13708,
    },
    "compared_to_notna": 3027,
}


def _value_counts(df: pd.DataFrame, column: str) -> dict:
    return df[column].value_counts(dropna=False).sort_index().to_dict()


def _result_stats(df: pd.DataFrame) -> dict:
    return {
        "tie_result": _value_counts(df, "tie_result"),
        "z_flag_homogenized": _value_counts(df, "z_flag_homogenized"),
        "instrument_type_homogenized": _value_counts(
            df, "instrument_type_homogenized"
        ),
        "compared_to_notna": int(df["compared_to"].notna().sum()),
    }


def _env_with_mamba_root_prefix() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("CRC_LOG_LEVEL", "INFO")

    if "MAMBA_ROOT_PREFIX" not in env:
        home = Path.home()
        for root in (home / "micromamba", home / ".micromamba", home / "miniconda3"):
            if (root / "envs" / ENV_NAME).exists():
                env["MAMBA_ROOT_PREFIX"] = str(root)
                break

    return env


def _temporary_config(tmp_path: Path) -> Path:
    with CONFIG_PATH.open() as fp:
        config = yaml.safe_load(fp)

    config["output_root_dir"] = str(tmp_path / "storage")
    config["output_dir"] = "outputs"
    config["param"]["flags_translation_file"] = str(REPO_ROOT / "flags_translation.yaml")

    config_path = tmp_path / "config.all.yaml"
    with config_path.open("w") as fp:
        yaml.safe_dump(config, fp, sort_keys=False)

    return config_path


@pytest.mark.slow
def test_all_test_data_pipeline_statistics(tmp_path):
    run_dir = tmp_path / "process_all"
    config_path = _temporary_config(tmp_path)

    completed = subprocess.run(
        [str(RUN_SCRIPT), str(config_path), str(run_dir)],
        cwd=REPO_ROOT,
        env=_env_with_mamba_root_prefix(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30 * 60,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout[-8000:]

    output_path = run_dir / "crd.parquet"
    assert output_path.exists(), f"Missing pipeline output: {output_path}"

    result = pd.read_parquet(output_path)
    assert _result_stats(result) == EXPECTED_STATS
