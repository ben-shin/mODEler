import numpy as np

from odefit.data.dataset import Dataset
from odefit.fitting.observable_vector import ObservableParameters
from odefit.simulation.simulation_result import SimulationResult


def predict_observable_values(
    simulation_result: SimulationResult,
    observable_parameters: ObservableParameters,
) -> dict[str, np.ndarray]:
    """
    Predict observed data columns from simulated species.

    Current observable model:

        predicted = scale * species + offset
    """

    predictions: dict[str, np.ndarray] = {}

    for data_column, observable in observable_parameters.items():
        species_name = str(observable["species"])
        scale = float(observable["scale"])
        offset = float(observable["offset"])

        species_values = simulation_result.get_species_values(species_name)

        predictions[data_column] = scale * species_values + offset

    return predictions


def calculate_observable_residuals(
    dataset: Dataset,
    simulation_result: SimulationResult,
    observable_parameters: ObservableParameters,
    use_normalized_data: bool = False,
) -> np.ndarray:
    """
    Calculate residuals using observable mappings.

    residual = predicted_observable - observed_data
    """

    predictions = predict_observable_values(
        simulation_result=simulation_result,
        observable_parameters=observable_parameters,
    )

    residual_blocks = []

    for data_column, predicted_values in predictions.items():
        observed_values = dataset.get_signal_values(
            signal_column=data_column,
            normalized=use_normalized_data,
        )

        if len(observed_values) != len(predicted_values):
            raise ValueError(
                f"Observed and predicted lengths do not match for {data_column}"
            )

        residual_blocks.append(predicted_values - observed_values)

    return np.concatenate(residual_blocks)
