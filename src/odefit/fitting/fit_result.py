from dataclasses import dataclass

import numpy as np

from odefit.simulation.simulation_result import SimulationResult


@dataclass
class FitResult:
    """
    Result of fitting a model to data.
    """

    success: bool
    message: str
    fitted_parameters: dict[str, float]
    initial_parameters: dict[str, float]
    residuals: np.ndarray
    statistics: dict[str, float]
    simulation_result: SimulationResult
    nfev: int
    cost: float
    fitted_initial_conditions: dict[str, float] | None = None
    initial_conditions: dict[str, float] | None = None
