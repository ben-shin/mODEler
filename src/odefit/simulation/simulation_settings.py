from dataclasses import dataclass


@dataclass
class SimulationSettings:
    """
    Settings for numerical ODE simulation
    """

    method: str = "RK45"
    rtol: float = 1e-6
    atol: float = 1e-9
