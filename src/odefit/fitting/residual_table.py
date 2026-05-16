import pandas as pd

from odefit.data.dataset import Dataset
from odefit.simulation.simulation_result import SimulationResult


def build_residual_table(
    dataset: Dataset,
    simulation_result: SimulationResult,
    species_mapping: dict[str, str],
    use_normalized_data: bool = False,
) -> pd.DataFrame:
    """
    Build a table containing observed values, fitted values, and residuals.

    residual = fitted - observed
    """

    output = pd.DataFrame(
        {
            dataset.time_column: dataset.time_values,
        }
    )

    for data_column, model_species in species_mapping.items():
        observed_values = dataset.get_signal_values(
            signal_column=data_column,
            normalized=use_normalized_data,
        )

        fitted_values = simulation_result.get_species_values(model_species)
        residual_values = fitted_values - observed_values

        output[f"{data_column}_observed"] = observed_values
        output[f"{data_column}_fit"] = fitted_values
        output[f"{data_column}_residual"] = residual_values

    return output
