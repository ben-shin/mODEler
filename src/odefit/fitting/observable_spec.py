from dataclasses import dataclass


@dataclass
class ObservableSpec:
    """
    Specification for mapping a model species to an observed data column.

    The current supported observable model is:

        observed = scale * species + offset

    scale and offset can each be fixed or fitted.
    """

    data_column: str
    species: str

    scale_initial_guess: float = 1.0
    scale_lower_bound: float = 0.0
    scale_upper_bound: float = float("inf")
    scale_fixed: bool = True
    scale_fixed_value: float | None = 1.0

    offset_initial_guess: float = 0.0
    offset_lower_bound: float = -float("inf")
    offset_upper_bound: float = float("inf")
    offset_fixed: bool = True
    offset_fixed_value: float | None = 0.0
