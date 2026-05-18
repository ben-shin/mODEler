import pandas as pd

from odefit.data.dataset import Dataset
from odefit.fitting.observable_residuals import predict_observable_values
from odefit.fitting.observable_vector import ObservableParameters
from odefit.simulation.simulation_result import SimulationResult


def build_observable_residual_table(
    dataset: Dataset,
    simulation_result: SimulationResult,
    observable_parameters: ObservableParameters,
    use_normalized_data: bool = False,
) -> pd.DataFrame:
    """
    Build residual table for observable mappings.

    Current observable model:

        predicted = scale * species + offset

    Output columns:

        time
        signal_observed
        signal_fit
        signal_residual

    where:

        residual = predicted - observed
    """

    predictions = predict_observable_values(
        simulation_result=simulation_result,
        observable_parameters=observable_parameters,
    )

    table_data = {
        dataset.time_column: dataset.time_values,
    }

    for data_column, predicted_values in predictions.items():
        observed_values = dataset.get_signal_values(
            signal_column=data_column,
            normalized=use_normalized_data,
        )

        if len(observed_values) != len(predicted_values):
            raise ValueError(
                f"Observed and predicted lengths do not match for {data_column}"
            )

        residual_values = predicted_values - observed_values

        table_data[f"{data_column}_observed"] = observed_values
        table_data[f"{data_column}_fit"] = predicted_values
        table_data[f"{data_column}_residual"] = residual_values

    return pd.DataFrame(table_data)
