from pathlib import Path

import matplotlib.pyplot as plt

from odefit.data.dataset import Dataset
from odefit.plotting.plot_settings import PlotSettings
from odefit.simulation.simulation_result import SimulationResult


def validate_axis_scale(scale: str) -> None:
    """
    Validate a matplotlib axis scale.
    """

    allowed_scales = {"linear", "log"}

    if scale not in allowed_scales:
        raise ValueError(f"Invalid axis scale: {scale}")


def apply_plot_settings(ax, settings: PlotSettings) -> None:
    """
    Apply common plot settings to a matplotlib Axes object.
    """

    validate_axis_scale(settings.x_scale)
    validate_axis_scale(settings.y_scale)

    ax.set_xscale(settings.x_scale)
    ax.set_yscale(settings.y_scale)

    ax.set_xlabel(settings.x_label)
    ax.set_ylabel(settings.y_label)

    if settings.title is not None:
        ax.set_title(settings.title)

    if settings.show_grid:
        ax.grid(True)

    if settings.show_legend:
        ax.legend()


def plot_dataset_timecourse(
    dataset: Dataset,
    signal_columns: list[str] | None = None,
    normalized: bool = False,
    settings: PlotSettings | None = None,
):
    """
    Plot experimental signal columns over time.
    """

    if settings is None:
        settings = PlotSettings(
            x_label=dataset.time_column,
            y_label="Signal",
        )

    if signal_columns is None:
        signal_columns = dataset.signal_columns

    fig, ax = plt.subplots()

    time_values = dataset.time_values

    for signal_column in signal_columns:
        signal_values = dataset.get_signal_values(
            signal_column=signal_column,
            normalized=normalized,
        )

        ax.scatter(
            time_values,
            signal_values,
            label=signal_column,
        )

    apply_plot_settings(ax, settings)

    return fig, ax


def plot_simulation_timecourse(
    simulation_result: SimulationResult,
    species: list[str] | None = None,
    settings: PlotSettings | None = None,
):
    """
    Plot simulated species concentrations over time.
    """

    if settings is None:
        settings = PlotSettings(
            x_label="Time",
            y_label="Concentration",
        )

    if species is None:
        species = simulation_result.species

    fig, ax = plt.subplots()

    for species_name in species:
        values = simulation_result.get_species_values(species_name)

        ax.plot(
            simulation_result.timepoints,
            values,
            label=species_name,
        )

    apply_plot_settings(ax, settings)

    return fig, ax


def save_figure(fig, file_path: str | Path) -> Path:
    """
    Save a matplotlib figure to disk.
    """

    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(path, bbox_inches="tight")

    return path
