import numpy as np
from scipy.integrate import solve_ivp

from odefit.model.model_spec import ModelSpec
from odefit.simulation.rhs import (
    build_rhs_function,
    validate_initial_conditions,
    validate_parameter_values,
)
from odefit.simulation.simulation_result import SimulationResult
from odefit.simulation.simulation_settings import SimulationSettings

SUPPORTED_SOLVER_METHODS = {
    "RK45",
    "RK23",
    "DOP853",
    "Radau",
    "BDF",
    "LSODA",
}


def validate_timepoints(
    timepoints: np.ndarray,
) -> None:
    """
    Validate simulation timepoints.
    """

    if len(timepoints) < 2:
        raise ValueError("At least two timepoints are required")

    differences = np.diff(timepoints)

    if np.any(differences <= 0):
        raise ValueError("Timepoints must be strictly increasing")


def validate_solver_method(
    method: str,
) -> None:
    """
    Validate solve_ivp method name.
    """

    if method not in SUPPORTED_SOLVER_METHODS:
        allowed = ", ".join(sorted(SUPPORTED_SOLVER_METHODS))
        raise ValueError(f"Unsupported solver method: {method}. Allowed: {allowed}")


def build_initial_value_vector(
    model: ModelSpec,
    initial_conditions: dict[str, float],
) -> np.ndarray:
    """
    Build initial value vector in model species order.
    """

    return np.array(
        [initial_conditions[species_name] for species_name in model.species],
        dtype=float,
    )


def detect_negative_value_warnings(
    values: np.ndarray,
    species: list[str],
    tolerance: float,
) -> list[str]:
    """
    Detect negative simulated values.

    Only values below -tolerance are reported.
    """

    warnings: list[str] = []

    for species_index, species_name in enumerate(species):
        species_values = values[:, species_index]

        minimum_value = float(np.min(species_values))

        if minimum_value < -tolerance:
            warnings.append(
                f"Species {species_name} reached negative simulated value "
                f"{minimum_value}"
            )

    return warnings


def simulate_model(
    model: ModelSpec,
    parameters: dict[str, float],
    initial_conditions: dict[str, float],
    timepoints,
    settings: SimulationSettings | None = None,
) -> SimulationResult:
    """
    Simulate a model using scipy.integrate.solve_ivp.
    """

    if settings is None:
        settings = SimulationSettings()

    validate_solver_method(settings.method)

    timepoints_array = np.asarray(timepoints, dtype=float)

    validate_timepoints(timepoints_array)

    validate_parameter_values(
        model=model,
        parameters=parameters,
    )

    validate_initial_conditions(
        model=model,
        initial_conditions=initial_conditions,
    )

    initial_values = build_initial_value_vector(
        model=model,
        initial_conditions=initial_conditions,
    )

    rhs = build_rhs_function(
        model=model,
        parameters=parameters,
        clip_negative_concentrations=settings.clip_negative_concentrations,
    )

    solution = solve_ivp(
        fun=rhs,
        t_span=(float(timepoints_array[0]), float(timepoints_array[-1])),
        y0=initial_values,
        t_eval=timepoints_array,
        method=settings.method,
        rtol=settings.rtol,
        atol=settings.atol,
    )

    values = solution.y.T

    warnings: list[str] = []

    if settings.warn_on_negative_values:
        warnings.extend(
            detect_negative_value_warnings(
                values=values,
                species=model.species,
                tolerance=settings.negative_warning_tolerance,
            )
        )

    if not solution.success:
        warnings.append(f"ODE solver failed: {solution.message}")

    return SimulationResult(
        timepoints=timepoints_array,
        species=model.species,
        values=values,
        success=bool(solution.success),
        message=str(solution.message),
        warnings=warnings,
    )
