import numpy as np

from odefit.data.dataset import Dataset
from odefit.simulation.simulation_result import SimulationResult


def calculate_residuals(
    dataset: Dataset,
    simulation_result: SimulationResult,
    species_mapping: dict[str, str],
    use_normalized_data: bool = False,
) -> np.ndarray:
    """
    Calculate residual vector.

    species_mapping maps data columns to model species.

    residual = simulation - observed
    """

    residual_blocks = []

    for data_column, model_species in species_mapping.items():
        observed_values = dataset.get_signal_values(
            signal_column=data_column,
            normalized=use_normalized_data,
        )

        simulated_values = simulation_result.get_species_values(model_species)

        if len(observed_values) != len(simulated_values):
            raise ValueError(
                f"Observed and simulated lengths do not match for {data_column}"
            )

        residual_blocks.append(simulated_values - observed_values)

    return np.concatenate(residual_blocks)
