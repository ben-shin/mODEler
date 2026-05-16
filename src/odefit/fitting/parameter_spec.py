from dataclasses import dataclass


@dataclass
class ParameterSpec:
    """
    Specifications for one fitted or fixed parameter
    """

    name: str
    initial_guess: float
    lower_bound: float = 0.0
    upper_bound: float = float("inf")
    fixed: bool = False
    fixed_value: float | None = None
