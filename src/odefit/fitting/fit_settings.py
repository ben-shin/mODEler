from dataclasses import dataclass


@dataclass
class FitSettings:
    """
    Settings controlling model fitting.
    """

    species_mapping: dict[str, str]
    use_normalized_data: bool = False

    method: str = "trf"
    loss: str = "linear"
    max_nfev: int | None = None

    rtol: float = 1e-6
    atol: float = 1e-9

    # Optional per-signal residual weights.
    #
    # weighted_residual = signal_weight * (predicted - observed)
    #
    # Example:
    #     {"amide_percent": 2.0}
    signal_weights: dict[str, float] | None = None
