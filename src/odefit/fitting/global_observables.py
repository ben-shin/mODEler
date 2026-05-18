from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from odefit.data.dataset import Dataset
from odefit.fitting.fit_result import FitResult
from odefit.fitting.fit_settings import FitSettings
from odefit.fitting.initial_condition_spec import InitialConditionSpec
from odefit.fitting.observable_spec import ObservableSpec
from odefit.fitting.optimizer import fit_model
from odefit.fitting.parameter_spec import ParameterSpec
from odefit.model.model_spec import ModelSpec


@dataclass
class GlobalObservableFitOutput:
    """
    Output from global observable fitting.

    fit_result:
        The normal FitResult object returned by fit_model.

    observable_specs:
        The ObservableSpec objects used for the fit. These are useful for
        export tables and later GUI display.
    """

    fit_result: FitResult
    observable_specs: list[ObservableSpec]


def infer_signal_columns_from_dataframe(
    dataframe: pd.DataFrame,
    time_column: str,
    exclude_columns: list[str] | None = None,
    numeric_only: bool = True,
) -> list[str]:
    """
    Infer signal columns from a wide-format dataframe.

    The time column is excluded automatically.

    Extra metadata columns can be excluded with exclude_columns.

    If numeric_only is True, non-numeric columns are skipped.

    This is useful for HSQC peak-intensity tables like:

        time,A23_HN,G45_HN,L78_HN
        0,1000,850,1200
        1,920,810,1105
    """

    if time_column not in dataframe.columns:
        raise ValueError(f"Time column not found in dataframe: {time_column}")

    excluded = {time_column}

    if exclude_columns is not None:
        excluded.update(exclude_columns)

    signal_columns: list[str] = []

    for column in dataframe.columns:
        if column in excluded:
            continue

        if numeric_only and not pd.api.types.is_numeric_dtype(dataframe[column]):
            continue

        signal_columns.append(column)

    if not signal_columns:
        raise ValueError("No signal columns could be inferred")

    return signal_columns


def read_wide_observable_dataset(
    file_path: str | Path,
    time_column: str = "time",
    signal_columns: list[str] | None = None,
    exclude_columns: list[str] | None = None,
    numeric_only: bool = True,
) -> Dataset:
    """
    Read a wide-format observable dataset.

    If signal_columns is None, signal columns are inferred.

    This is the recommended first data format for assigned HSQC peaks:

        time,A23_HN,G45_HN,L78_HN
        0,1000,850,1200
        1,920,810,1105

    The returned Dataset contains all peak columns as signal columns.
    """

    path = Path(file_path)

    dataframe = pd.read_csv(path)

    if signal_columns is None:
        signal_columns = infer_signal_columns_from_dataframe(
            dataframe=dataframe,
            time_column=time_column,
            exclude_columns=exclude_columns,
            numeric_only=numeric_only,
        )

    return Dataset(
        raw_dataframe=dataframe,
        time_column=time_column,
        signal_columns=signal_columns,
    )


def build_shared_species_observable_specs(
    signal_columns: list[str],
    species: str,
    fit_scale: bool = True,
    fit_offset: bool = True,
    scale_initial_guess: float = 1.0,
    scale_lower_bound: float = 0.0,
    scale_upper_bound: float = float("inf"),
    offset_initial_guess: float = 0.0,
    offset_lower_bound: float = -float("inf"),
    offset_upper_bound: float = float("inf"),
) -> list[ObservableSpec]:
    """
    Build one ObservableSpec per signal column.

    Each signal column is mapped to the same model species:

        signal_i = scale_i * species + offset_i

    This is the core global HSQC use case:

        peak_A23 = scale_A23 * A + offset_A23
        peak_G45 = scale_G45 * A + offset_G45
        peak_L78 = scale_L78 * A + offset_L78

    The kinetic parameters controlling A(t) are shared globally.
    The scale/offset values are peak-specific.
    """

    if not signal_columns:
        raise ValueError("At least one signal column is required")

    observable_specs: list[ObservableSpec] = []

    for signal_column in signal_columns:
        observable_specs.append(
            ObservableSpec(
                data_column=signal_column,
                species=species,
                scale_initial_guess=scale_initial_guess,
                scale_lower_bound=scale_lower_bound,
                scale_upper_bound=scale_upper_bound,
                scale_fixed=not fit_scale,
                scale_fixed_value=scale_initial_guess if not fit_scale else None,
                offset_initial_guess=offset_initial_guess,
                offset_lower_bound=offset_lower_bound,
                offset_upper_bound=offset_upper_bound,
                offset_fixed=not fit_offset,
                offset_fixed_value=offset_initial_guess if not fit_offset else None,
            )
        )

    return observable_specs


def build_signal_weights_from_columns(
    signal_columns: list[str],
    default_weight: float = 1.0,
) -> dict[str, float]:
    """
    Build a simple per-signal weight dictionary.

    This is useful when all peaks should initially be weighted equally.

    More advanced weighting, for example based on noise or replicate standard
    deviation, can be added later.
    """

    if default_weight <= 0:
        raise ValueError("default_weight must be positive")

    return {signal_column: float(default_weight) for signal_column in signal_columns}


def fit_global_observable_model(
    model: ModelSpec,
    dataset: Dataset,
    parameter_specs: list[ParameterSpec],
    initial_condition_specs: list[InitialConditionSpec],
    observed_species: str,
    settings: FitSettings | None = None,
    signal_columns: list[str] | None = None,
    observable_specs: list[ObservableSpec] | None = None,
    fit_scale: bool = True,
    fit_offset: bool = True,
    scale_initial_guess: float = 1.0,
    scale_lower_bound: float = 0.0,
    scale_upper_bound: float = float("inf"),
    offset_initial_guess: float = 0.0,
    offset_lower_bound: float = -float("inf"),
    offset_upper_bound: float = float("inf"),
) -> GlobalObservableFitOutput:
    """
    Fit many observed signals to one global kinetic model.

    Shared globally:
        - kinetic parameters
        - model species timecourses
        - initial conditions, unless the user explicitly fits them

    Peak/signal-specific:
        - scale
        - offset

    Default observable model:

        signal_i(t) = scale_i * observed_species(t) + offset_i

    Example for HSQC peaks:

        A23_HN(t) = scale_A23 * A(t) + offset_A23
        G45_HN(t) = scale_G45 * A(t) + offset_G45
        L78_HN(t) = scale_L78 * A(t) + offset_L78
    """

    if observed_species not in model.species:
        raise ValueError(f"Observed species is not in model: {observed_species}")

    if signal_columns is None:
        signal_columns = dataset.signal_columns

    if observable_specs is None:
        observable_specs = build_shared_species_observable_specs(
            signal_columns=signal_columns,
            species=observed_species,
            fit_scale=fit_scale,
            fit_offset=fit_offset,
            scale_initial_guess=scale_initial_guess,
            scale_lower_bound=scale_lower_bound,
            scale_upper_bound=scale_upper_bound,
            offset_initial_guess=offset_initial_guess,
            offset_lower_bound=offset_lower_bound,
            offset_upper_bound=offset_upper_bound,
        )

    if settings is None:
        settings = FitSettings(
            species_mapping={},
        )

    fit_result = fit_model(
        model=model,
        dataset=dataset,
        parameter_specs=parameter_specs,
        initial_condition_specs=initial_condition_specs,
        observable_specs=observable_specs,
        settings=settings,
    )

    return GlobalObservableFitOutput(
        fit_result=fit_result,
        observable_specs=observable_specs,
    )
