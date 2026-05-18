import numpy as np

from odefit.data.dataset import Dataset
from odefit.fitting.observable_spec import ObservableSpec
from odefit.model.model_spec import ModelSpec

ObservableParameters = dict[str, dict[str, float | str]]


def make_observable_specs_from_species_mapping(
    species_mapping: dict[str, str],
) -> list[ObservableSpec]:
    """
    Convert old-style species_mapping into fixed direct observable mappings.

    This keeps the old API behavior:

        data column = model species
    """

    return [
        ObservableSpec(
            data_column=data_column,
            species=model_species,
            scale_fixed=True,
            scale_fixed_value=1.0,
            offset_fixed=True,
            offset_fixed_value=0.0,
        )
        for data_column, model_species in species_mapping.items()
    ]


def get_free_observable_parameter_names(
    observable_specs: list[ObservableSpec],
) -> list[str]:
    """
    Return names of observable parameters that should be optimized.
    """

    names: list[str] = []

    for observable in observable_specs:
        if not observable.scale_fixed:
            names.append(f"{observable.data_column}_scale")

        if not observable.offset_fixed:
            names.append(f"{observable.data_column}_offset")

    return names


def build_observable_vector(
    observable_specs: list[ObservableSpec],
) -> np.ndarray:
    """
    Build optimizer vector for free observable parameters.
    """

    values: list[float] = []

    for observable in observable_specs:
        if not observable.scale_fixed:
            values.append(observable.scale_initial_guess)

        if not observable.offset_fixed:
            values.append(observable.offset_initial_guess)

    return np.array(values, dtype=float)


def build_observable_bounds(
    observable_specs: list[ObservableSpec],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build optimizer bounds for free observable parameters.
    """

    lower_bounds: list[float] = []
    upper_bounds: list[float] = []

    for observable in observable_specs:
        if not observable.scale_fixed:
            lower_bounds.append(observable.scale_lower_bound)
            upper_bounds.append(observable.scale_upper_bound)

        if not observable.offset_fixed:
            lower_bounds.append(observable.offset_lower_bound)
            upper_bounds.append(observable.offset_upper_bound)

    return np.array(lower_bounds, dtype=float), np.array(upper_bounds, dtype=float)


def get_fixed_scale_value(observable: ObservableSpec) -> float:
    """
    Return fixed scale value.

    If scale_fixed_value is None, scale_initial_guess is used.
    """

    if observable.scale_fixed_value is not None:
        return observable.scale_fixed_value

    return observable.scale_initial_guess


def get_fixed_offset_value(observable: ObservableSpec) -> float:
    """
    Return fixed offset value.

    If offset_fixed_value is None, offset_initial_guess is used.
    """

    if observable.offset_fixed_value is not None:
        return observable.offset_fixed_value

    return observable.offset_initial_guess


def vector_to_observable_parameters(
    vector: np.ndarray,
    observable_specs: list[ObservableSpec],
) -> ObservableParameters:
    """
    Convert optimizer vector into full observable parameters.

    Output structure:

        {
            "amide": {
                "species": "A",
                "scale": 2.0,
                "offset": 0.1,
            }
        }
    """

    observable_parameters: ObservableParameters = {}
    vector_index = 0

    for observable in observable_specs:
        if observable.scale_fixed:
            scale = get_fixed_scale_value(observable)
        else:
            if vector_index >= len(vector):
                raise ValueError("Observable vector is shorter than expected")

            scale = float(vector[vector_index])
            vector_index += 1

        if observable.offset_fixed:
            offset = get_fixed_offset_value(observable)
        else:
            if vector_index >= len(vector):
                raise ValueError("Observable vector is shorter than expected")

            offset = float(vector[vector_index])
            vector_index += 1

        observable_parameters[observable.data_column] = {
            "species": observable.species,
            "scale": scale,
            "offset": offset,
        }

    if vector_index != len(vector):
        raise ValueError("Observable vector is longer than expected")

    return observable_parameters


def build_initial_observable_parameters(
    observable_specs: list[ObservableSpec],
) -> ObservableParameters:
    """
    Build observable parameters from initial guesses/fixed values.
    """

    vector = build_observable_vector(observable_specs)

    return vector_to_observable_parameters(
        vector=vector,
        observable_specs=observable_specs,
    )


def validate_observable_specs(
    model: ModelSpec,
    dataset: Dataset,
    observable_specs: list[ObservableSpec],
) -> None:
    """
    Validate observable specifications.
    """

    if not observable_specs:
        raise ValueError("At least one ObservableSpec is required")

    seen_data_columns = set()

    for observable in observable_specs:
        if observable.data_column in seen_data_columns:
            raise ValueError(
                f"Duplicate observable data column: {observable.data_column}"
            )

        seen_data_columns.add(observable.data_column)

        if observable.data_column not in dataset.signal_columns:
            raise ValueError(
                f"Observable data column is not in dataset: {observable.data_column}"
            )

        if observable.species not in model.species:
            raise ValueError(
                f"Observable species is not in model: {observable.species}"
            )

        if observable.scale_lower_bound > observable.scale_upper_bound:
            raise ValueError(
                f"Scale lower bound exceeds upper bound for {observable.data_column}"
            )

        if observable.offset_lower_bound > observable.offset_upper_bound:
            raise ValueError(
                f"Offset lower bound exceeds upper bound for {observable.data_column}"
            )

        if not (
            observable.scale_lower_bound
            <= observable.scale_initial_guess
            <= observable.scale_upper_bound
        ):
            raise ValueError(
                f"Scale initial guess outside bounds for {observable.data_column}"
            )

        if not (
            observable.offset_lower_bound
            <= observable.offset_initial_guess
            <= observable.offset_upper_bound
        ):
            raise ValueError(
                f"Offset initial guess outside bounds for {observable.data_column}"
            )

        if observable.scale_fixed:
            scale_value = get_fixed_scale_value(observable)

            if not (
                observable.scale_lower_bound
                <= scale_value
                <= observable.scale_upper_bound
            ):
                raise ValueError(
                    f"Fixed scale outside bounds for {observable.data_column}"
                )

        if observable.offset_fixed:
            offset_value = get_fixed_offset_value(observable)

            if not (
                observable.offset_lower_bound
                <= offset_value
                <= observable.offset_upper_bound
            ):
                raise ValueError(
                    f"Fixed offset outside bounds for {observable.data_column}"
                )
