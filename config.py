import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator

MAINDIR = Path(__file__).parent
DATASETS_DIR = os.getenv("DATASETS_DIR", "/datasets")
DASK_EXECUTOR = os.getenv("DASK_EXECUTOR", "local")


class Slurm(BaseModel):
    class Instance(BaseModel):
        cores: int = 2
        processes: int = 1
        memory: str = "20GB"
        queue: str = "cpu_pipelines"
        account: str = "hpc-pipelines"
        job_extra_directives: list[str] = ["--propagate", "--time=12:00:00"]

    class Scale(BaseModel):
        minimum_jobs: int = 3
        maximum_jobs: int = 22
        adaptive_interval_seconds: float = 10.0
        adaptive_scale_down_delay_seconds: float = 180.0
        worker_recovery_timeout_seconds: float = 360.0
        worker_recovery_check_interval_seconds: float = 10.0

        @model_validator(mode="after")
        def validate_limits(self):
            if self.minimum_jobs < 1:
                raise ValueError("minimum_jobs must be at least 1")
            if self.maximum_jobs < self.minimum_jobs:
                raise ValueError("maximum_jobs must be >= minimum_jobs")
            if self.adaptive_interval_seconds <= 0:
                raise ValueError("adaptive_interval_seconds must be positive")
            if self.adaptive_scale_down_delay_seconds <= 0:
                raise ValueError(
                    "adaptive_scale_down_delay_seconds must be positive"
                )
            if self.worker_recovery_timeout_seconds <= 0:
                raise ValueError("worker_recovery_timeout_seconds must be positive")
            if self.worker_recovery_check_interval_seconds <= 0:
                raise ValueError(
                    "worker_recovery_check_interval_seconds must be positive"
                )
            return self

    instance: Instance = Instance()
    scale: Scale = Scale()


class Local(BaseModel):
    n_workers: int = 2
    threads_per_worker: int = 2
    memory_limit: str = "1GiB"


class Executor(BaseModel):
    name: str = DASK_EXECUTOR
    args: Any = {}

    @model_validator(mode="before")
    @classmethod
    def sync_args(cls, data: Any) -> Any:

        assert isinstance(data, dict), "data is not dict"
        name = data.get("name", DASK_EXECUTOR)

        match name:
            case "local":
                executor = Local(**data.get("args", {}))
            case "slurm":
                executor = Slurm(**data.get("args", {}))
            case _:
                raise ValueError(f"name '{name}' do not match")

        data["args"] = executor.model_dump()
        return data


class Inputs(BaseModel):
    class Specz(BaseModel):
        class Columns(BaseModel):
            id: str | None = None
            ra: str = "ra"
            dec: str = "dec"
            z: str = "z"
            z_flag: str | None = None
            z_err: str | None = None
            survey: str | None = None

        path: str = f"{DATASETS_DIR}/specz.parquet"
        internal_name: str = "00_specz"
        format: str = "parquet"
        columns: Columns = Columns()

    specz: list = [Specz(), Specz()]


class Param(BaseModel):
    class Run(BaseModel):
        combine_type: str = "concatenate"
        tie_treatment_option: str = "remove_all"
        flags_translation_file: str = str(Path(MAINDIR, "flags_translation.yaml"))

    class Preparation(BaseModel):
        repartition_prepared_catalogs: bool = False
        prepared_partition_size: str = "256MB"

    class Diagnostics(BaseModel):
        tie_invariant_diagnostics_enabled: bool = True
        tie_invariant_diagnostics_detailed_enabled: bool = False
        tie_invariant_diagnostics_sample_size: int = 10
        tie_invariant_diagnostics_max_rows: int = 100
        label_merge_diagnostics_enabled: bool = True
        crossmatch_geometry_diagnostics_enabled: bool = False
        representative_radius_diagnostics_enabled: bool = False
        dedup_edge_diagnostics_enabled: bool = False
        save_expr_columns: bool = False

        @model_validator(mode="after")
        def validate_limits(self):
            if self.tie_invariant_diagnostics_sample_size < 1:
                raise ValueError(
                    "tie_invariant_diagnostics_sample_size must be positive"
                )
            if self.tie_invariant_diagnostics_max_rows < 1:
                raise ValueError("tie_invariant_diagnostics_max_rows must be positive")
            return self

    class Filters(BaseModel):
        class InstrumentTypeHomogenized(BaseModel):
            include_spectroscopic: bool = True
            include_grism: bool = True
            include_photometric: bool = True
            include_unclassified: bool = True

            @model_validator(mode="after")
            def validate_any_enabled(self):
                if not any(self.model_dump().values()):
                    raise ValueError(
                        "at least one instrument_type_homogenized include option "
                        "must be true"
                    )
                return self

        class ObjectTypeHomogenized(BaseModel):
            include_unclassified: bool = True
            include_galaxy: bool = True
            include_star: bool = False
            include_agn: bool = True
            include_qso: bool = True
            include_galactic: bool = False

            @model_validator(mode="after")
            def validate_any_enabled(self):
                if not any(self.model_dump().values()):
                    raise ValueError(
                        "at least one object_type_homogenized include option "
                        "must be true"
                    )
                return self

        # Zero disables the cut; valid active cuts are 1, 2, 3, 4.
        z_flag_homogenized_value_to_cut: float = 3.0
        instrument_type_homogenized: InstrumentTypeHomogenized = (
            InstrumentTypeHomogenized()
        )
        object_type_homogenized: ObjectTypeHomogenized = ObjectTypeHomogenized()

    class Output(BaseModel):
        class HomogenizedColumns(BaseModel):
            z_flag_homogenized: str = "always"
            instrument_type_homogenized: str = "always"
            object_type_homogenized: str = "always"

            @model_validator(mode="after")
            def validate_modes(self):
                valid = {"auto", "always", "never"}
                for field, value in self.model_dump().items():
                    if value not in valid:
                        raise ValueError(
                            f"output.homogenized_columns.{field} must be one of "
                            f"{sorted(valid)}"
                        )
                return self

        extra_columns: dict[str, Any] = Field(default_factory=dict)
        homogenized_columns: HomogenizedColumns = HomogenizedColumns()
        insert_DP1_footprint_flag: bool = False
        insert_rubin_footprint_flag: bool = False

    run: Run = Run()
    filters: Filters = Filters()
    preparation: Preparation = Preparation()
    diagnostics: Diagnostics = Diagnostics()
    output: Output = Output()

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_layout(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        migrated = dict(data)

        run = dict(migrated.get("run") or {})
        for key in ("combine_type", "tie_treatment_option", "flags_translation_file"):
            if key in migrated and key not in run:
                run[key] = migrated[key]
        if run:
            migrated["run"] = run

        filters = dict(migrated.get("filters") or {})
        if "z_flag_homogenized_value_to_cut" in migrated:
            filters.setdefault(
                "z_flag_homogenized_value_to_cut",
                migrated["z_flag_homogenized_value_to_cut"],
            )

        instrument = dict(filters.get("instrument_type_homogenized") or {})
        instrument_aliases = {
            "include_spectroscopic_ith": "include_spectroscopic",
            "include_grism_ith": "include_grism",
            "include_photometric_ith": "include_photometric",
            "include_unclassified_ith": "include_unclassified",
        }
        for old, new in instrument_aliases.items():
            if old in migrated and new not in instrument:
                instrument[new] = migrated[old]
        if instrument:
            filters["instrument_type_homogenized"] = instrument

        object_type = dict(filters.get("object_type_homogenized") or {})
        object_aliases = {
            "include_unclassified_oth": "include_unclassified",
            "include_galaxy_oth": "include_galaxy",
            "include_star_oth": "include_star",
            "include_agn_oth": "include_agn",
            "include_qso_oth": "include_qso",
            "include_galactic_oth": "include_galactic",
        }
        for old, new in object_aliases.items():
            if old in migrated and new not in object_type:
                object_type[new] = migrated[old]
        if object_type:
            filters["object_type_homogenized"] = object_type
        if filters:
            migrated["filters"] = filters

        output = dict(migrated.get("output") or {})
        if "extra_columns" in migrated:
            output.setdefault("extra_columns", migrated["extra_columns"])
        if "output_homogenized_columns" in migrated:
            output.setdefault(
                "homogenized_columns", migrated["output_homogenized_columns"]
            )
        for key in ("insert_DP1_footprint_flag", "insert_rubin_footprint_flag"):
            if key in migrated and key not in output:
                output[key] = migrated[key]
        if output:
            migrated["output"] = output

        diagnostics = dict(migrated.get("diagnostics") or {})
        diagnostics.pop("expr_column_schema", None)
        if diagnostics:
            migrated["diagnostics"] = diagnostics

        return migrated


class Config(BaseModel):
    output_root_dir: str = "."
    output_dir: str = "outputs"
    output_format: str | None = "parquet"
    output_name: str = "crd"
    executor: Executor = Executor()
    inputs: Inputs = Inputs()
    param: Param = Param()


if __name__ == "__main__":
    import yaml

    cfg = Config()
    data = cfg.model_dump()

    with open("config.yml", "w") as outfile:
        print(cfg.model_dump_json(indent=2))
        yaml.dump(data, outfile, sort_keys=False, allow_unicode=True)
