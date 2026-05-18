from dataclasses import dataclass


@dataclass
class SimulationSettings:
    """
    Settings for numerical ODE simulation.
    """

    method: str = "LSODA"
    rtol: float = 1e-6
    atol: float = 1e-9

    # If True, negative values passed into the RHS are clipped to zero
    # before calculating reaction rates.
    #
    # This does not forcibly clip the final solve_ivp output. It only prevents
    # small numerical negative values from producing unphysical mass-action rates.
    clip_negative_concentrations: bool = False

    # If True, the returned SimulationResult will contain warnings if any
    # simulated values are negative below negative_warning_tolerance.
    warn_on_negative_values: bool = True

    negative_warning_tolerance: float = 1e-9
