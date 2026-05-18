from pathlib import Path

import matplotlib.pyplot as plt
from odefit.plotting.plot_settings import PlotSettings
from odefit.plotting.timecourse_plots import apply_plot_settings

from odefit.data.dataset import Dataset
from odefit.fitting.residual_table import build_residual_table
from odefit.simulation.simulation_result import SimulationResult


def plot_residuals_over_time(
    dataset: Dataset,
    simulation_result: SimulationResult,
    species_mapping: dict[str, str],
    use_normalized_data: bool = False,
    settings: PlotSettings | None = None,
):
    """
    Plot residuals over time.

    residual = fitted - observed
    """

    if settings is None:
        settings = PlotSettings(
            title="Residuals over time",
            x_label=dataset.time_column,
            y_label="Residual",
        )

    residual_table = build_residual_table(
        dataset=dataset,
        simulation_result=simulation_result,
        species_mapping=species_mapping,
        use_normalized_data=use_normalized_data,
    )

    fig, ax = plt.subplots()

    time_values = residual_table[dataset.time_column]

    for data_column in species_mapping:
        residual_column = f"{data_column}_residual"

        ax.axhline(
            y=0.0,
            linewidth=1,
        )

        ax.scatter(
            time_values,
            residual_table[residual_column],
            label=f"{data_column} residual",
        )

    apply_plot_settings(ax, settings)

    return fig, ax


def save_residuals_plot(
    dataset: Dataset,
    simulation_result: SimulationResult,
    species_mapping: dict[str, str],
    file_path: str | Path,
    use_normalized_data: bool = False,
    settings: PlotSettings | None = None,
) -> Path:
    """
    Create and save a residual plot.
    """

    fig, _ = plot_residuals_over_time(
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
