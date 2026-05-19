from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from odefit.data.dataset import Dataset
from odefit.fitting.fit_result import FitResult
from odefit.fitting.fit_settings import FitSettings
from odefit.fitting.initial_condition_spec import InitialConditionSpec
from odefit.fitting.observable_spec import ObservableSpec
from odefit.fitting.parameter_spec import ParameterSpec
from odefit.model.model_spec import ModelSpec


def to_jsonable(value: Any) -> Any:
    """
    Convert common Python/numpy/pandas/dataclass objects into JSON-safe values.
    """

    if value is None:
        return None

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, pd.Index):
        return value.tolist()

    if isinstance(value, pd.Series):
        return value.tolist()

    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")

    if is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(asdict(value))

    if isinstance(value, dict):
        return {
            str(key): to_jsonable(item)
            for key, item in value.items()
        }

    if isinstance(value, tuple):
        return [
            to_jsonable(item)
            for item in value
        ]

    if isinstance(value, list):
        return [
            to_jsonable(item)
            for item in value
        ]

    if isinstance(value, set):
        return sorted(
            to_jsonable(item)
            for item in value
        )

    if isinstance(value, bool | int | float | str):
        return value

    return str(value)


def write_json(
    data: dict,
    file_path: str | Path,
) -> Path:
    """
    Write JSON data to file.
    """

    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    json_data = to_jsonable(data)

    path.write_text(
        json.dumps(
            json_data,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    return path


def build_model_metadata(
    model: ModelSpec,
) -> dict:
    """
    Build JSON-safe model metadata.
    """

    return {
        "name": model.name,
        "metadata": model.metadata,
        "raw_text": model.raw_text,
        "species": model.species,
        "parameters": model.parameters,
        "warnings": model.warnings,
        "reactions": [
            {
                "index": index,
                "label": reaction.label,
                "source_line": reaction.source_line,
                "reactants": reaction.reactants,
                "products": reaction.products,
                "reversible": reaction.reversible,
                "forward_rate": reaction.forward_rate,
                "reverse_rate": reaction.reverse_rate,
            }
            for index, reaction in enumerate(model.reactions, start=1)
        ],
    }


def build_dataset_metadata(
    dataset: Dataset,
) -> dict:
    """
    Build JSON-safe dataset metadata.
    """

    dataframe = dataset.raw_dataframe

    return {
        "time_column": dataset.time_column,
        "signal_columns": dataset.signal_columns,
        "normalization_method": dataset.normalization_method,
        "n_rows": int(len(dataframe)),
        "n_columns": int(len(dataframe.columns)),
        "columns": list(dataframe.columns),
    }


def build_parameter_specs_metadata(
    parameter_specs: list[ParameterSpec] | None,
) -> list[dict]:
    """
    Build JSON-safe parameter spec metadata.
    """

    if parameter_specs is None:
        return []

    return [
        to_jsonable(parameter_spec)
        for parameter_spec in parameter_specs
    ]


def build_initial_condition_specs_metadata(
    initial_condition_specs: list[InitialConditionSpec] | None,
) -> list[dict]:
    """
    Build JSON-safe initial condition spec metadata.
    """

    if initial_condition_specs is None:
        return []

    return [
        to_jsonable(initial_condition_spec)
        for initial_condition_spec in initial_condition_specs
    ]


def build_observable_specs_metadata(
    observable_specs: list[ObservableSpec] | None,
) -> list[dict]:
    """
    Build JSON-safe observable spec metadata.
    """

    if observable_specs is None:
        return []

    return [
        to_jsonable(observable_spec)
        for observable_spec in observable_specs
    ]


def build_fit_settings_metadata(
    fit_settings: FitSettings | None,
) -> dict:
    """
    Build JSON-safe fit settings metadata.
    """

    if fit_settings is None:
        return {}

    return to_jsonable(fit_settings)


def build_fit_result_summary(
    fit_result: FitResult | None,
) -> dict:
    """
    Build compact JSON-safe fit result summary.

    This deliberately does not dump the full residual vector or simulation
    matrix because those are already exported as CSV.
    """

    if fit_result is None:
        return {}

    return {
        "success": fit_result.success,
        "message": fit_result.message,
        "fitted_parameters": fit_result.fitted_parameters,
        "initial_parameters": fit_result.initial_parameters,
        "fitted_initial_conditions": fit_result.fitted_initial_conditions,
        "initial_conditions": fit_result.initial_conditions,
        "fitted_observables": fit_result.fitted_observables,
        "statistics": fit_result.statistics,
        "nfev": fit_result.nfev,
        "cost": fit_result.cost,
        "optimizer_status": fit_result.optimizer_status,
        "optimizer_optimality": fit_result.optimizer_optimality,
        "optimizer_active_mask": fit_result.optimizer_active_mask,
        "optimizer_njev": fit_result.optimizer_njev,
    }


def build_run_metadata(
    command: str | None = None,
    config_path: str | Path | None = None,
    extra_metadata: dict | None = None,
) -> dict:
    """
    Build reproducibility metadata for a run.
    """

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "config_path": None if config_path is None else str(config_path),
        "python_version": sys.version,
        "platform": platform.platform(),
        "platform_system": platform.system(),
        "platform_release": platform.release(),
    }

    if extra_metadata is not None:
        metadata["extra_metadata"] = extra_metadata

    return metadata


def export_fit_metadata_json(
    output_dir: str | Path,
    model: ModelSpec,
    dataset: Dataset | None = None,
    fit_result: FitResult | None = None,
    fit_settings: FitSettings | None = None,
    parameter_specs: list[ParameterSpec] | None = None,
    initial_condition_specs: list[InitialConditionSpec] | None = None,
    observable_specs: list[ObservableSpec] | None = None,
    command: str | None = None,
    config_path: str | Path | None = None,
    extra_run_metadata: dict | None = None,
) -> dict[str, Path]:
    """
    Export JSON metadata files for a fit or simulation run.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    written_files: dict[str, Path] = {}

    written_files["model_metadata_json"] = write_json(
        data=build_model_metadata(model),
        file_path=output_path / "model_metadata.json",
    )

    if dataset is not None:
        written_files["dataset_metadata_json"] = write_json(
            data=build_dataset_metadata(dataset),
            file_path=output_path / "dataset_metadata.json",
        )

    written_files["parameter_specs_json"] = write_json(
        data={
            "parameter_specs": build_parameter_specs_metadata(parameter_specs),
        },
        file_path=output_path / "parameter_specs.json",
    )

    written_files["initial_condition_specs_json"] = write_json(
        data={
            "initial_condition_specs": build_initial_condition_specs_metadata(
                initial_condition_specs
            ),
        },
        file_path=output_path / "initial_condition_specs.json",
    )

    written_files["observable_specs_json"] = write_json(
        data={
            "observable_specs": build_observable_specs_metadata(observable_specs),
        },
        file_path=output_path / "observable_specs.json",
    )

    written_files["fit_settings_json"] = write_json(
        data={
            "fit_settings": build_fit_settings_metadata(fit_settings),
        },
        file_path=output_path / "fit_settings.json",
    )

    if fit_result is not None:
        written_files["fit_result_summary_json"] = write_json(
            data=build_fit_result_summary(fit_result),
            file_path=output_path / "fit_result_summary.json",
        )

    written_files["run_metadata_json"] = write_json(
        data=build_run_metadata(
            command=command,
            config_path=config_path,
            extra_metadata=extra_run_metadata,
        ),
        file_path=output_path / "run_metadata.json",
    )

    return written_files
