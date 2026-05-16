from dataclasses import dataclass


@dataclass
class ParameterSpec:
    """
    Specification for one kinetic parameter.

    Parameters can be:
    - free: optimized by least_squares
    - fixed: held at fixed_value
    - tied: forced to equal another parameter
    """

    name: str
    initial_guess: float
    lower_bound: float = 0.0
    upper_bound: float = float("inf")
    fixed: bool = False
    fixed_value: float | None = None
    tied_to: str | None = None
