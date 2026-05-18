from dataclasses import dataclass


@dataclass
class FitSettings:
    """
    Setting controling model fitting
    """

    species_mapping: dict[str, str]
    use_normalized_data: bool = False
    max_nfev: int | None = None
    loss: str = "linear"
    method: str = "trf"
    rtol: float = 1e-6
    atol: float = 1e-9
