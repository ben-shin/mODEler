from pathlib import Path

import matplotlib.pyplot as plt

from odefit.data.dataset import Dataset
from odefit.fitting.observable_residuals import predict_observable_values
from odefit.fitting.observable_vector import ObservableParameters
from odefit.plotting.plot_settings import PlotSettings
from odefit.plotting.timecourse_plots import apply_plot_settings
from odefit.simulation.simulation_result import SimulationResult


def plot_observed_vs_predicted_observables(
    dataset: Dataset,
    simulation_result: SimulationResult,
    observable_parameters: ObservableParameters,
    use_normalized_data: bool = False,
    settings: PlotSettings | None = None,
):
    """
    Plot observed data against predicted observable values over time.

    Predicted observable model:

        predicted = scale * species + offset

    This is the correct plot to use when the experimental signal is not
    directly equal to a model species concentration.
    """

    if settings is None:
        settings = PlotSettings(
            title="Observed vs fitted observables",
            x_label=dataset.time_column,
            y_label="Signal",
        )

    predictions = predict_observable_values(
        simulation_result=simulation_result,
        observable_parameters=observable_parameters,
    )

    fig, ax = plt.subplots()

    time_values = dataset.time_values

    for data_column, predicted_values in predictions.items():
        observed_values = dataset.get_signal_values(
            signal_column=data_column,
            normalized=use_normalized_data,
        )

        ax.scatter(
            time_values,
            observed_values,
            label=f"{data_column} observed",
        )

        ax.plot(
            simulation_result.timepoints,
            predicted_values,
            label=f"{data_column} fitted",
        )

    apply_plot_settings(ax, settings)

    return fig, ax


def plot_observable_residuals_over_time(
    dataset: Dataset,
    simulation_result: SimulationResult,
    observable_parameters: ObservableParameters,
    use_normalized_data: bool = False,
    settings: PlotSettings | None = None,
):
    """
    Plot observable residuals over time.

    residual = predicted observable - observed data
    """

    if settings is None:
        settings = PlotSettings(
            title="Observable residuals over time",
            x_label=dataset.time_column,
            y_label="Residual",
        )

    predictions = predict_observable_values(
        simulation_result=simulation_result,
        observable_parameters=observable_parameters,
    )

    fig, ax = plt.subplots()

    time_values = dataset.time_values

    ax.axhline(
        y=0.0,
        linewidth=1,
    )

    for data_column, predicted_values in predictions.items():
        observed_values = dataset.get_signal_values(
            signal_column=data_column,
            normalized=use_normalized_data,
        )

        residual_values = predicted_values - observed_values

        ax.scatter(
            time_values,
            residual_values,
            label=f"{data_column} residual",
        )

    apply_plot_settings(ax, settings)

    return fig, ax


def save_observed_vs_predicted_observables_plot(
    dataset: Dataset,
    simulation_result: SimulationResult,
    observable_parameters: ObservableParameters,
    file_path: str | Path,
    use_normalized_data: bool = False,
    settings: PlotSettings | None = None,
) -> Path:
    """
    Save observed-vs-predicted observable plot.
    """

    fig, _ = plot_observed_vs_predicted_observables(
        dataset=dataset,
        simulation_result=simulation_result,
        observable_parameters=observable_parameters,
        use_normalized_data=use_normalized_data,
        settings=settings,
    )

    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

    return path


def save_observable_residuals_plot(
    dataset: Dataset,
    simulation_result: SimulationResult,
    observable_parameters: ObservableParameters,
    file_path: str | Path,
    use_normalized_data: bool = False,
    settings: PlotSettings | None = None,
) -> Path:
    """
    Save observable residual plot.
    """

    fig, _ = plot_observable_residuals_over_time(
        dataset=dataset,
        simulation_result=simulation_result,
        observable_parameters=observable_parameters,
        use_normalized_data=use_normalized_data,
        settings=settings,
    )

    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

    return path
