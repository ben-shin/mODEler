import numpy as np
from scipy.integrate import solve_ivp

from odefit.model.model_spec import ModelSpec
from odefit.simulation.rhs import build_rhs_function
from odefit.simulation.simulation_result import SimulationResult
from odefit.simulation.simulation_settings import SimulationSettings


def validate_simulation_inputs(
    model: ModelSpec,
    parameters: dict[str, float],
    initial_conditions: dict[str, float],
    timepoints: list[float] | np.ndarray,
) -> None:
    """
    Validate simulation inputs before calling the ODE solver.
    """

    for parameter_name in model.parameters:
        if parameter_name not in parameters:
            raise ValueError(f"Missing parameter value for {parameter_name}")

    for species_name in model.species:
        if species_name not in initial_conditions:
            raise ValueError(f"Missing initial condition for {species_name}")

    if len(timepoints) < 2:
        raise ValueError("At least two timepoints are required for simulation")

    timepoints_array = np.asarray(timepoints, dtype=float)

    if not np.all(np.diff(timepoints_array) > 0):
        raise ValueError("Timepoints must be strictly increasing")


def simulate_model(
    model: ModelSpec,
    parameters: dict[str, float],
    initial_conditions: dict[str, float],
    timepoints: list[float] | np.ndarray,
    settings: SimulationSettings | None = None,
) -> SimulationResult:
    """
    Simulate a ModelSpec over specified timepoints.
    """

    if settings is None:
        settings = SimulationSettings()

    validate_simulation_inputs(
        model=model,
        parameters=parameters,
        initial_conditions=initial_conditions,
        timepoints=timepoints,
    )

    timepoints_array = np.asarray(timepoints, dtype=float)

    y0 = np.array(
        [initial_conditions[species_name] for species_name in model.species],
        dtype=float,
    )

    rhs = build_rhs_function(
        model=model,
        parameters=parameters,
    )

    solution = solve_ivp(
        fun=rhs,
        t_span=(timepoints_array[0], timepoints_array[-1]),
        y0=y0,
        t_eval=timepoints_array,
        method=settings.method,
        rtol=settings.rtol,
        atol=settings.atol,
    )

    if not solution.success:
        raise RuntimeError(f"ODE solver failed: {solution.message}")

    values = solution.y.T

    return SimulationResult(
        timepoints=timepoints_array,
        species=model.species,
        values=values,
    )
