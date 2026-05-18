from pathlib import Path

import matplotlib.pyplot as plt
from odefit.plotting.plot_settings import PlotSettings
from odefit.plotting.timecourse_plots import apply_plot_settings

from odefit.data.dataset import Dataset
from odefit.simulation.simulation_result import SimulationResult


def plot_observed_vs_fitted_timecourse(
    dataset: Dataset,
    simulation_result: SimulationResult,
    species_mapping: dict[str, str],
    use_normalized_data: bool = False,
    settings: PlotSettings | None = None,
):
    """
    Plot observed data points against fitted simulation curves over time.

    species_mapping maps data columns to model species.

    Example:
        {
            "amide": "A",
        }
    """

    if settings is None:
        settings = PlotSettings(
            title="Observed vs fitted",
            x_label=dataset.time_column,
            y_label="Signal",
        )

    fig, ax = plt.subplots()

    time_values = dataset.time_values

    for data_column, model_species in species_mapping.items():
        observed_values = dataset.get_signal_values(
            signal_column=data_column,
            normalized=use_normalized_data,
        )

        fitted_values = simulation_result.get_species_values(model_species)

        ax.scatter(
            time_values,
            observed_values,
            label=f"{data_column} observed",
        )

        ax.plot(
            simulation_result.timepoints,
            fitted_values,
            label=f"{model_species} fitted",
        )

    apply_plot_settings(ax, settings)

    return fig, ax


def save_observed_vs_fitted_plot(
    dataset: Dataset,
    simulation_result: SimulationResult,
    species_mapping: dict[str, str],
    file_path: str | Path,
    use_normalized_data: bool = False,
    settings: PlotSettings | None = None,
) -> Path:
    """
    Create and save an observed-vs-fitted timecourse plot.
    """

    fig, _ = plot_observed_vs_fitted_timecourse(
        dataset=dataset,
        simulation_result=simulation_result,
        species_mapping=species_mapping,
        use_normalized_data=use_normalized_data,
        settings=settings,
    )

    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

    return path
