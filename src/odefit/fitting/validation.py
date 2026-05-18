from odefit.data.dataset import Dataset
from odefit.fitting.fit_settings import FitSettings
from odefit.fitting.initial_condition_spec import InitialConditionSpec
from odefit.fitting.initial_condition_vector import (
    get_free_initial_condition_specs,
    make_fixed_initial_condition_specs,
    validate_initial_condition_specs,
)
from odefit.fitting.observable_spec import ObservableSpec
from odefit.fitting.observable_vector import (
    get_free_observable_parameter_names,
    validate_observable_specs,
)
from odefit.fitting.parameter_spec import ParameterSpec
from odefit.fitting.parameter_vector import (
    get_free_parameter_specs,
    validate_parameter_specs,
)
from odefit.model.model_spec import ModelSpec


def resolve_initial_condition_specs(
    initial_conditions: dict[str, float] | None = None,
    initial_condition_specs: list[InitialConditionSpec] | None = None,
) -> list[InitialConditionSpec]:
    """
    Resolve old-style initial_conditions or new-style initial_condition_specs.

    Only one should be provided.
    """

    if initial_conditions is not None and initial_condition_specs is not None:
        raise ValueError(
            "Provide either initial_conditions or initial_condition_specs, not both"
        )

    if initial_condition_specs is not None:
        return initial_condition_specs

    if initial_conditions is not None:
        return make_fixed_initial_condition_specs(initial_conditions)

    raise ValueError("Initial conditions are required")


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


def validate_model_initial_condition_specs(
    model: ModelSpec,
    initial_condition_specs: list[InitialConditionSpec],
) -> None:
    """
    Validate that initial-condition specs match model species exactly.
    """

    model_species = set(model.species)
    spec_species = {
        initial_condition.species for initial_condition in initial_condition_specs
    }

    missing_species = model_species - spec_species
    extra_species = spec_species - model_species

    if missing_species:
        missing = ", ".join(sorted(missing_species))
        raise ValueError(f"Missing InitialConditionSpec entries for: {missing}")

    if extra_species:
        extra = ", ".join(sorted(extra_species))
        raise ValueError(f"InitialConditionSpec entries not present in model: {extra}")


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
    number_of_mapped_signals: int,
    number_of_free_variables: int,
) -> None:
    """
    Validate that there are more residuals than free variables.
    """

    number_of_timepoints = len(dataset.time_values)
    number_of_residuals = number_of_timepoints * number_of_mapped_signals

    if number_of_free_variables == 0:
        raise ValueError("At least one free variable is required for fitting")

    if number_of_residuals <= number_of_free_variables:
        raise ValueError(
            "Number of residuals must be greater than number of free variables"
        )


def validate_signal_weights(
    dataset: Dataset,
    settings: FitSettings,
    mapped_data_columns: list[str],
) -> None:
    """
    Validate optional per-signal residual weights.
    """

    if settings.signal_weights is None:
        return

    mapped_data_column_set = set(mapped_data_columns)

    for data_column, weight in settings.signal_weights.items():
        if data_column not in dataset.signal_columns:
            raise ValueError(
                f"Signal weight provided for unknown data column: {data_column}"
            )

        if data_column not in mapped_data_column_set:
            raise ValueError(
                f"Signal weight provided for unmapped data column: {data_column}"
            )

        if weight <= 0:
            raise ValueError(
                f"Signal weight must be positive for data column: {data_column}"
            )


def validate_fit_inputs(
    model: ModelSpec,
    dataset: Dataset,
    parameter_specs: list[ParameterSpec],
    initial_conditions: dict[str, float] | None = None,
    settings: FitSettings | None = None,
    initial_condition_specs: list[InitialConditionSpec] | None = None,
    observable_specs: list[ObservableSpec] | None = None,
) -> list[InitialConditionSpec]:
    """
    Validate all fitting inputs before optimization.

    Returns resolved InitialConditionSpec objects.
    """

    if settings is None:
        raise ValueError("FitSettings are required")

    resolved_initial_condition_specs = resolve_initial_condition_specs(
        initial_conditions=initial_conditions,
        initial_condition_specs=initial_condition_specs,
    )

    validate_parameter_specs(parameter_specs)
    validate_initial_condition_specs(resolved_initial_condition_specs)

    validate_model_parameter_specs(model, parameter_specs)
    validate_model_initial_condition_specs(model, resolved_initial_condition_specs)

    if observable_specs is not None:
        validate_observable_specs(
            model=model,
            dataset=dataset,
            observable_specs=observable_specs,
        )

        mapped_data_columns = [
            observable.data_column for observable in observable_specs
        ]

        number_of_mapped_signals = len(observable_specs)
        number_of_free_observable_parameters = len(
            get_free_observable_parameter_names(observable_specs)
        )

    else:
        validate_species_mapping(model, dataset, settings)

        mapped_data_columns = list(settings.species_mapping.keys())

        number_of_mapped_signals = len(settings.species_mapping)
        number_of_free_observable_parameters = 0

    validate_signal_weights(
        dataset=dataset,
        settings=settings,
        mapped_data_columns=mapped_data_columns,
    )
    validate_normalized_data_available(dataset, settings)

    number_of_free_parameters = len(get_free_parameter_specs(parameter_specs))
    number_of_free_initial_conditions = len(
        get_free_initial_condition_specs(resolved_initial_condition_specs)
    )

    validate_residual_count(
        dataset=dataset,
        number_of_mapped_signals=number_of_mapped_signals,
        number_of_free_variables=(
            number_of_free_parameters
            + number_of_free_initial_conditions
            + number_of_free_observable_parameters
        ),
    )

    return resolved_initial_condition_specs
