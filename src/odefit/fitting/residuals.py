import numpy as np

from odefit.data.dataset import Dataset
from odefit.simulation.simulation_result import SimulationResult


def get_signal_weight(
    data_column: str,
    signal_weights: dict[str, float] | None = None,
) -> float:
    """
    Get residual weight for one data column.

    If no weight is provided, use 1.0.
    """

    if signal_weights is None:
        return 1.0

    return float(signal_weights.get(data_column, 1.0))


def calculate_residuals(
    dataset: Dataset,
    simulation_result: SimulationResult,
    species_mapping: dict[str, str],
    use_normalized_data: bool = False,
    signal_weights: dict[str, float] | None = None,
) -> np.ndarray:
    """
    Calculate residuals between observed data and simulated species.

    residual = fitted - observed

    If signal_weights are provided:

        residual = weight * (fitted - observed)
    """

    residual_blocks = []

    for data_column, model_species in species_mapping.items():
        observed_values = dataset.get_signal_values(
            signal_column=data_column,
            normalized=use_normalized_data,
        )

        fitted_values = simulation_result.get_species_values(model_species)

        if len(observed_values) != len(fitted_values):
            raise ValueError(
                f"Observed and simulated lengths do not match for {data_column}"
            )

        residual_values = fitted_values - observed_values

        weight = get_signal_weight(
            data_column=data_column,
            signal_weights=signal_weights,
        )

        residual_blocks.append(weight * residual_values)

    return np.concatenate(residual_blocks)
