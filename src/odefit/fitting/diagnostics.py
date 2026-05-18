import math

import pandas as pd

from odefit.fitting.fit_result import FitResult
from odefit.fitting.initial_condition_spec import InitialConditionSpec
from odefit.fitting.observable_spec import ObservableSpec
from odefit.fitting.observable_vector import ObservableParameters
from odefit.fitting.parameter_spec import ParameterSpec

DIAGNOSTIC_COLUMNS = [
    "kind",
    "name",
    "value",
    "lower_bound",
    "upper_bound",
    "fixed",
    "tied_to",
    "optimized",
    "at_lower_bound",
    "at_upper_bound",
    "warning",
]


def is_at_lower_bound(
    value: float,
    lower_bound: float,
    tolerance: float = 1e-8,
) -> bool:
    """
    Return True if value is close to or below the lower bound.
    """

    if math.isinf(lower_bound) and lower_bound < 0:
        return False

    return value <= lower_bound + tolerance


def is_at_upper_bound(
    value: float,
    upper_bound: float,
    tolerance: float = 1e-8,
) -> bool:
    """
    Return True if value is close to or above the upper bound.
    """

    if math.isinf(upper_bound) and upper_bound > 0:
        return False

    return value >= upper_bound - tolerance


def make_bound_warning(
    at_lower_bound: bool,
    at_upper_bound: bool,
) -> str:
    """
    Create a warning string for bound diagnostics.
    """

    if at_lower_bound and at_upper_bound:
        return "at_lower_and_upper_bound"

    if at_lower_bound:
        return "at_lower_bound"

    if at_upper_bound:
        return "at_upper_bound"

    return ""


def build_parameter_diagnostics_table(
    parameter_specs: list[ParameterSpec],
    fitted_parameters: dict[str, float],
    tolerance: float = 1e-8,
) -> pd.DataFrame:
    """
    Build diagnostics for kinetic parameters.

    Only free, independently optimized parameters are flagged for bound warnings.
    Fixed and tied parameters are shown, but are not treated as optimizer-bound issues.
    """

    rows = []

    for parameter in parameter_specs:
        if parameter.name not in fitted_parameters:
            raise ValueError(f"Missing fitted parameter: {parameter.name}")

        value = float(fitted_parameters[parameter.name])

        optimized = not parameter.fixed and parameter.tied_to is None

        at_lower = optimized and is_at_lower_bound(
            value=value,
            lower_bound=parameter.lower_bound,
            tolerance=tolerance,
        )

        at_upper = optimized and is_at_upper_bound(
            value=value,
            upper_bound=parameter.upper_bound,
            tolerance=tolerance,
        )

        rows.append(
            {
                "kind": "parameter",
                "name": parameter.name,
                "value": value,
                "lower_bound": parameter.lower_bound,
                "upper_bound": parameter.upper_bound,
                "fixed": parameter.fixed,
                "tied_to": parameter.tied_to,
                "optimized": optimized,
                "at_lower_bound": at_lower,
                "at_upper_bound": at_upper,
                "warning": make_bound_warning(at_lower, at_upper),
            }
        )

    return pd.DataFrame(rows, columns=DIAGNOSTIC_COLUMNS)


def build_initial_condition_diagnostics_table(
    initial_condition_specs: list[InitialConditionSpec],
    fitted_initial_conditions: dict[str, float],
    tolerance: float = 1e-8,
) -> pd.DataFrame:
    """
    Build diagnostics for fitted initial conditions.
    """

    rows = []

    for initial_condition in initial_condition_specs:
        species = initial_condition.species

        if species not in fitted_initial_conditions:
            raise ValueError(f"Missing fitted initial condition: {species}")

        value = float(fitted_initial_conditions[species])

        optimized = not initial_condition.fixed

        at_lower = optimized and is_at_lower_bound(
            value=value,
            lower_bound=initial_condition.lower_bound,
            tolerance=tolerance,
        )

        at_upper = optimized and is_at_upper_bound(
            value=value,
            upper_bound=initial_condition.upper_bound,
            tolerance=tolerance,
        )

        rows.append(
            {
                "kind": "initial_condition",
                "name": species,
                "value": value,
                "lower_bound": initial_condition.lower_bound,
                "upper_bound": initial_condition.upper_bound,
                "fixed": initial_condition.fixed,
                "tied_to": None,
                "optimized": optimized,
                "at_lower_bound": at_lower,
                "at_upper_bound": at_upper,
                "warning": make_bound_warning(at_lower, at_upper),
            }
        )

    return pd.DataFrame(rows, columns=DIAGNOSTIC_COLUMNS)


def build_observable_diagnostics_table(
    observable_specs: list[ObservableSpec],
    fitted_observables: ObservableParameters,
    tolerance: float = 1e-8,
) -> pd.DataFrame:
    """
    Build diagnostics for observable scale/offset parameters.
    """

    rows = []

    for observable in observable_specs:
        data_column = observable.data_column

        if data_column not in fitted_observables:
            raise ValueError(f"Missing fitted observable: {data_column}")

        fitted_observable = fitted_observables[data_column]

        scale_value = float(fitted_observable["scale"])
        scale_optimized = not observable.scale_fixed

        scale_at_lower = scale_optimized and is_at_lower_bound(
            value=scale_value,
            lower_bound=observable.scale_lower_bound,
            tolerance=tolerance,
        )

        scale_at_upper = scale_optimized and is_at_upper_bound(
            value=scale_value,
            upper_bound=observable.scale_upper_bound,
            tolerance=tolerance,
        )

        rows.append(
            {
                "kind": "observable_scale",
                "name": f"{data_column}_scale",
                "value": scale_value,
                "lower_bound": observable.scale_lower_bound,
                "upper_bound": observable.scale_upper_bound,
                "fixed": observable.scale_fixed,
                "tied_to": None,
                "optimized": scale_optimized,
                "at_lower_bound": scale_at_lower,
                "at_upper_bound": scale_at_upper,
                "warning": make_bound_warning(scale_at_lower, scale_at_upper),
            }
        )

        offset_value = float(fitted_observable["offset"])
        offset_optimized = not observable.offset_fixed

        offset_at_lower = offset_optimized and is_at_lower_bound(
            value=offset_value,
            lower_bound=observable.offset_lower_bound,
            tolerance=tolerance,
        )

        offset_at_upper = offset_optimized and is_at_upper_bound(
            value=offset_value,
            upper_bound=observable.offset_upper_bound,
            tolerance=tolerance,
        )

        rows.append(
            {
                "kind": "observable_offset",
                "name": f"{data_column}_offset",
                "value": offset_value,
                "lower_bound": observable.offset_lower_bound,
                "upper_bound": observable.offset_upper_bound,
                "fixed": observable.offset_fixed,
                "tied_to": None,
                "optimized": offset_optimized,
                "at_lower_bound": offset_at_lower,
                "at_upper_bound": offset_at_upper,
                "warning": make_bound_warning(offset_at_lower, offset_at_upper),
            }
        )

    return pd.DataFrame(rows, columns=DIAGNOSTIC_COLUMNS)


def build_fit_diagnostics_table(
    fit_result: FitResult,
    parameter_specs: list[ParameterSpec],
    initial_condition_specs: list[InitialConditionSpec] | None = None,
    observable_specs: list[ObservableSpec] | None = None,
    tolerance: float = 1e-8,
) -> pd.DataFrame:
    """
    Build one combined diagnostics table for a fit result.
    """

    tables = [
        build_parameter_diagnostics_table(
            parameter_specs=parameter_specs,
            fitted_parameters=fit_result.fitted_parameters,
            tolerance=tolerance,
        )
    ]

    if initial_condition_specs is not None:
        if fit_result.fitted_initial_conditions is None:
            raise ValueError("FitResult is missing fitted initial conditions")

        tables.append(
            build_initial_condition_diagnostics_table(
                initial_condition_specs=initial_condition_specs,
                fitted_initial_conditions=fit_result.fitted_initial_conditions,
                tolerance=tolerance,
            )
        )

    if observable_specs is not None:
        if fit_result.fitted_observables is None:
            raise ValueError("FitResult is missing fitted observables")

        tables.append(
            build_observable_diagnostics_table(
                observable_specs=observable_specs,
                fitted_observables=fit_result.fitted_observables,
                tolerance=tolerance,
            )
        )

    return pd.concat(tables, ignore_index=True)


def get_diagnostic_warnings(
    diagnostics_table: pd.DataFrame,
) -> list[str]:
    """
    Return human-readable diagnostic warnings.
    """

    warnings = []

    for _, row in diagnostics_table.iterrows():
        warning = row["warning"]

        if warning:
            warnings.append(
                f"{row['kind']} '{row['name']}' is {warning}: "
                f"value={row['value']}, "
                f"bounds=[{row['lower_bound']}, {row['upper_bound']}]"
            )

    return warnings
