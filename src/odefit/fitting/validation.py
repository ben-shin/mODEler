from odefit.data.dataset import Dataset
from odefit.fitting.fit_settings import FitSettings
from odefit.fitting.parameter_spec import ParameterSpec
from odefit.fitting.parameter_vector import (
    get_free_parameter_specs,
    validate_parameter_specs,
)
from odefit.model.model_spec import ModelSpec


def validate_model_parameter_specs(
    model: ModelSpec,
    parameter_specs: list[ParameterSpec],
) -> None:
    """
    Validate that parameter specs match model parameters exactly.
    """

    model_parameter_names = set(model.parameters)
    spec_parameter_names = {parameter.name for parameter in parameter_specs}

    missing_parameters = model_parameter_names - spec_parameter_names
    extra_parameters = spec_parameter_names - model_parameter_names

    if missing_parameters:
        missing = ", ".join(sorted(missing_parameters))
        raise ValueError(f"Missing ParameterSpec entries for: {missing}")

    if extra_parameters:
        extra = ", ".join(sorted(extra_parameters))
        raise ValueError(f"ParameterSpec entries not present in model: {extra}")


def validate_species_mapping(
    model: ModelSpec,
    dataset: Dataset,
    settings: FitSettings,
) -> None:
    """
    Validate data-column to model-species mapping.
    """

    if not settings.species_mapping:
        raise ValueError("species_mapping cannot be empty")

    for data_column, model_species in settings.species_mapping.items():
        if data_column not in dataset.signal_columns:
            raise ValueError(f"Mapped data column is not in dataset: {data_column}")

        if model_species not in model.species:
            raise ValueError(f"Mapped model species is not in model: {model_species}")


def validate_initial_conditions(
    model: ModelSpec,
    initial_conditions: dict[str, float],
) -> None:
    """
    Validate that all model species have initial conditions.
    """

    for species_name in model.species:
        if species_name not in initial_conditions:
            raise ValueError(f"Missing initial condition for species: {species_name}")


def validate_normalized_data_available(
    dataset: Dataset,
    settings: FitSettings,
) -> None:
    """
    Validate normalized data availability if normalized fitting is requested.
    """

    if settings.use_normalized_data and dataset.normalized_dataframe is None:
        raise ValueError("Normalized fitting requested, but normalized data is missing")


def validate_residual_count(
    dataset: Dataset,
    settings: FitSettings,
    number_of_free_parameters: int,
) -> None:
    """
    Validate that there are more residuals than free parameters.
    """

    number_of_timepoints = len(dataset.time_values)
    number_of_mapped_signals = len(settings.species_mapping)
    number_of_residuals = number_of_timepoints * number_of_mapped_signals

    if number_of_free_parameters == 0:
        raise ValueError("At least one free parameter is required for fitting")

    if number_of_residuals <= number_of_free_parameters:
        raise ValueError(
            "Number of residuals must be greater than number of free parameters"
        )


def validate_fit_inputs(
    model: ModelSpec,
    dataset: Dataset,
    parameter_specs: list[ParameterSpec],
    initial_conditions: dict[str, float],
    settings: FitSettings,
) -> None:
    """
    Validate all fitting inputs before optimization.
    """

    validate_parameter_specs(parameter_specs)
    validate_model_parameter_specs(model, parameter_specs)
    validate_species_mapping(model, dataset, settings)
    validate_initial_conditions(model, initial_conditions)
    validate_normalized_data_available(dataset, settings)

    number_of_free_parameters = len(get_free_parameter_specs(parameter_specs))

    validate_residual_count(
        dataset=dataset,
        settings=settings,
        number_of_free_parameters=number_of_free_parameters,
    )
