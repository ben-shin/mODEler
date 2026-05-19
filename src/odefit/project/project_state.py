from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from odefit.fitting.fit_settings import FitSettings
from odefit.fitting.initial_condition_spec import InitialConditionSpec
from odefit.fitting.observable_spec import ObservableSpec
from odefit.fitting.parameter_spec import ParameterSpec


PROJECT_STATE_SCHEMA_VERSION = 1


@dataclass
class ProjectState:
    """
    Serializable project state for GUI/project save-load.

    This stores the user's modelling setup, not the complete fit result.

    Fit results should be exported separately using export_fit_bundle().
    """

    schema_version: int = PROJECT_STATE_SCHEMA_VERSION

    project_name: str | None = None
    project_notes: str | None = None

    model_text: str = ""

    data_path: str | None = None
    time_column: str | None = None
    signal_columns: list[str] = field(default_factory=list)

    normalization_method: str | None = None

    species_mapping: dict[str, str] = field(default_factory=dict)

    parameter_specs: list[ParameterSpec] = field(default_factory=list)
    initial_condition_specs: list[InitialConditionSpec] = field(default_factory=list)
    observable_specs: list[ObservableSpec] = field(default_factory=list)

    fit_settings: FitSettings | None = None

    output_dir: str | None = None
    last_fit_output_dir: str | None = None

    metadata: dict[str, Any] = field(default_factory=dict)


def create_empty_project_state() -> ProjectState:
    """
    Create an empty project state.
    """

    return ProjectState()


def create_project_state(
    model_text: str,
    data_path: str | None = None,
    time_column: str | None = None,
    signal_columns: list[str] | None = None,
    normalization_method: str | None = None,
    species_mapping: dict[str, str] | None = None,
    parameter_specs: list[ParameterSpec] | None = None,
    initial_condition_specs: list[InitialConditionSpec] | None = None,
    observable_specs: list[ObservableSpec] | None = None,
    fit_settings: FitSettings | None = None,
    project_name: str | None = None,
    project_notes: str | None = None,
    output_dir: str | None = None,
    last_fit_output_dir: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> ProjectState:
    """
    Convenience constructor for a populated ProjectState.
    """

    return ProjectState(
        project_name=project_name,
        project_notes=project_notes,
        model_text=model_text,
        data_path=data_path,
        time_column=time_column,
        signal_columns=[] if signal_columns is None else signal_columns,
        normalization_method=normalization_method,
        species_mapping={} if species_mapping is None else species_mapping,
        parameter_specs=[] if parameter_specs is None else parameter_specs,
        initial_condition_specs=(
            []
            if initial_condition_specs is None
            else initial_condition_specs
        ),
        observable_specs=[] if observable_specs is None else observable_specs,
        fit_settings=fit_settings,
        output_dir=output_dir,
        last_fit_output_dir=last_fit_output_dir,
        metadata={} if metadata is None else metadata,
    )
