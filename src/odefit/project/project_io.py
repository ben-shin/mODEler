from __future__ import annotations

import json
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

from odefit.export.json_export import to_jsonable
from odefit.fitting.fit_settings import FitSettings
from odefit.fitting.initial_condition_spec import InitialConditionSpec
from odefit.fitting.observable_spec import ObservableSpec
from odefit.fitting.parameter_spec import ParameterSpec
from odefit.project.project_state import (
    PROJECT_STATE_SCHEMA_VERSION,
    ProjectState,
)


T = TypeVar("T")


def _dataclass_from_dict(
    cls: type[T],
    data: dict[str, Any],
) -> T:
    """
    Build a dataclass from a dictionary.

    Unknown keys are ignored. This makes project files more tolerant of
    older/newer schema versions.
    """

    if not is_dataclass(cls):
        raise TypeError(f"{cls} is not a dataclass")

    allowed_field_names = {
        field.name
        for field in fields(cls)
    }

    filtered_data = {
        key: value
        for key, value in data.items()
        if key in allowed_field_names
    }

    return cls(**filtered_data)


def parameter_spec_from_dict(
    data: dict[str, Any],
) -> ParameterSpec:
    """
    Restore a ParameterSpec from project JSON data.
    """

    return _dataclass_from_dict(ParameterSpec, data)


def initial_condition_spec_from_dict(
    data: dict[str, Any],
) -> InitialConditionSpec:
    """
    Restore an InitialConditionSpec from project JSON data.
    """

    return _dataclass_from_dict(InitialConditionSpec, data)


def observable_spec_from_dict(
    data: dict[str, Any],
) -> ObservableSpec:
    """
    Restore an ObservableSpec from project JSON data.
    """

    return _dataclass_from_dict(ObservableSpec, data)


def fit_settings_from_dict(
    data: dict[str, Any] | None,
) -> FitSettings | None:
    """
    Restore FitSettings from project JSON data.
    """

    if data is None:
        return None

    return _dataclass_from_dict(FitSettings, data)


def project_state_to_dict(
    project_state: ProjectState,
) -> dict[str, Any]:
    """
    Convert ProjectState to JSON-safe dictionary.
    """

    return to_jsonable(project_state)


def project_state_from_dict(
    data: dict[str, Any],
) -> ProjectState:
    """
    Restore ProjectState from dictionary data.
    """

    schema_version = int(
        data.get(
            "schema_version",
            PROJECT_STATE_SCHEMA_VERSION,
        )
    )

    if schema_version > PROJECT_STATE_SCHEMA_VERSION:
        raise ValueError(
            "Project file schema version is newer than this mODEler version: "
            f"{schema_version}"
        )

    parameter_specs = [
        parameter_spec_from_dict(parameter_data)
        for parameter_data in data.get("parameter_specs", [])
    ]

    initial_condition_specs = [
        initial_condition_spec_from_dict(initial_condition_data)
        for initial_condition_data in data.get("initial_condition_specs", [])
    ]

    observable_specs = [
        observable_spec_from_dict(observable_data)
        for observable_data in data.get("observable_specs", [])
    ]

    fit_settings = fit_settings_from_dict(
        data.get("fit_settings")
    )

    return ProjectState(
        schema_version=schema_version,
        project_name=data.get("project_name"),
        project_notes=data.get("project_notes"),
        model_text=data.get("model_text", ""),
        data_path=data.get("data_path"),
        time_column=data.get("time_column"),
        signal_columns=list(data.get("signal_columns", [])),
        normalization_method=data.get("normalization_method"),
        species_mapping=dict(data.get("species_mapping", {})),
        parameter_specs=parameter_specs,
        initial_condition_specs=initial_condition_specs,
        observable_specs=observable_specs,
        fit_settings=fit_settings,
        output_dir=data.get("output_dir"),
        last_fit_output_dir=data.get("last_fit_output_dir"),
        metadata=dict(data.get("metadata", {})),
    )


def save_project_state(
    project_state: ProjectState,
    file_path: str | Path,
) -> Path:
    """
    Save a ProjectState to a JSON project file.

    Suggested extension:
        .modeler.json
    """

    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = project_state_to_dict(project_state)

    path.write_text(
        json.dumps(
            data,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    return path


def load_project_state(
    file_path: str | Path,
) -> ProjectState:
    """
    Load a ProjectState from a JSON project file.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Project file does not exist: {path}")

    data = json.loads(path.read_text())

    return project_state_from_dict(data)


def validate_project_state_for_fitting(
    project_state: ProjectState,
) -> None:
    """
    Validate that a project state contains enough information to run a fit.

    This is intended for GUI preflight checks.
    """

    missing_fields: list[str] = []

    if not project_state.model_text:
        missing_fields.append("model_text")

    if not project_state.data_path:
        missing_fields.append("data_path")

    if not project_state.time_column:
        missing_fields.append("time_column")

    if not project_state.signal_columns:
        missing_fields.append("signal_columns")

    if not project_state.parameter_specs:
        missing_fields.append("parameter_specs")

    if not project_state.initial_condition_specs:
        missing_fields.append("initial_condition_specs")

    if project_state.fit_settings is None:
        missing_fields.append("fit_settings")

    if missing_fields:
        missing = ", ".join(missing_fields)
        raise ValueError(f"Project state is missing required fit fields: {missing}")


def validate_project_state_for_simulation(
    project_state: ProjectState,
) -> None:
    """
    Validate that a project state contains enough information to run simulation.

    This is intended for GUI preflight checks.
    """

    missing_fields: list[str] = []

    if not project_state.model_text:
        missing_fields.append("model_text")

    if not project_state.parameter_specs:
        missing_fields.append("parameter_specs")

    if not project_state.initial_condition_specs:
        missing_fields.append("initial_condition_specs")

    if missing_fields:
        missing = ", ".join(missing_fields)
        raise ValueError(
            f"Project state is missing required simulation fields: {missing}"
        )
