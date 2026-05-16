from dataclasses import dataclass


@dataclass
class InitialConditionSpec:
    """
    Specification for one initial species concentration/population.

    Initial conditions can be:
    - fixed: held constant during fitting
    - free: optimized by least_squares

    If fixed=True and fixed_value is None, initial_guess is used as the fixed value.
    """

    species: str
    initial_guess: float
    lower_bound: float = 0.0
    upper_bound: float = float("inf")
    fixed: bool = True
    fixed_value: float | None = None
