from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns # too sexy not to use

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
    Plot experimental variable columns over time.
    """

    sns.set_theme(style="whitegrid", context="paper")

    if settings is None:
        settings = PlotSettings(
            x_label=dataset.time_column,
            y_label="Signal",
        )

    if signal_columns is None:
        signal_columns = dataset.signal_columns

    assert signal_columns is not None

    if not signal_columns:
        fig, ax = plt.subplots()
        apply_plot_settings(ax, settings)
        return fig, ax

    validate_axis_scale(settings.x_scale)
    validate_axis_scale(settings.y_scale)

    mosaic_layout = [[col] for col in signal_columns]

    fig, axes_dict = plt.subplot_mosaic(
        mosaic_layout,
        sharex=True,
        figsize=(7, 2.5 * len(signal_columns))
    )

    time_values = dataset.time_values

    # Grab a nice Seaborn color palette
    palette = sns.color_palette("deep", n_colors=max(10, len(signal_columns)))

    for i, signal_column in enumerate(signal_columns):
        ax = axes_dict[signal_column]

        signal_values = dataset.get_signal_values(
            signal_column=signal_column,
            normalized=normalized,
        )

        # 2. Seaborn baby
        sns.scatterplot(
            x=time_values,
            y=signal_values,
            ax=ax,
            label=signal_column,
            marker="o",
            color=palette[i % len(palette)],
            s=55,  # slightly larger, crisp dots
            edgecolor="white",  # clean white border around dots
            linewidth=0.5
        )

        ax.set_xscale(settings.x_scale)
        ax.set_yscale(settings.y_scale)
        ax.set_ylabel(settings.y_label)
        ax.set_title(f"{signal_column}", fontsize=12, fontweight="600")

        if settings.show_legend:
            ax.legend()

        if i == len(signal_columns) - 1:
            ax.set_xlabel(settings.x_label)
        else:
            # Strip the X-label from upper plots to keep things perfectly clean
            ax.set_xlabel("")

    if settings.title is not None:
        fig.suptitle(settings.title, fontsize=15, fontweight="bold")

    fig.tight_layout()

    return fig, axes_dict


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


def plot_variable_comparison(
        datasets_dict: dict,
        variable: str,
        normalized: bool = False,
        settings: PlotSettings | None = None,
):
    """
    Plots the same variable across multiple datasets on a single axis.
    """
    sns.set_theme(style="whitegrid", context="paper")

    if settings is None:
        settings = PlotSettings(
            title=f"Comparison: {variable} across experiments",
            y_label="Concentration",
        )

    fig, ax = plt.subplots(figsize=(8, 5))

    validate_axis_scale(settings.x_scale)
    validate_axis_scale(settings.y_scale)

    # Dynamically generate enough colors for however many datasets you loaded
    palette = sns.color_palette("deep", n_colors=max(10, len(datasets_dict)))

    for i, (dataset_name, dataset) in enumerate(datasets_dict.items()):
        # Skip this dataset if it doesn't contain the variable we are comparing
        if variable not in dataset.signal_columns:
            continue

        time_values = dataset.time_values
        signal_values = dataset.get_signal_values(
            signal_column=variable,
            normalized=normalized,
        )

        sns.scatterplot(
            x=time_values,
            y=signal_values,
            ax=ax,
            label=dataset_name,
            marker="o",
            color=palette[i % len(palette)],
            s=65,
            edgecolor="white",
            linewidth=0.5
        )

    ax.set_xscale(settings.x_scale)
    ax.set_yscale(settings.y_scale)
    ax.set_ylabel(settings.y_label)
    ax.set_xlabel(settings.x_label)

    if settings.title is not None:
        ax.set_title(settings.title, fontsize=14, fontweight="bold")

    if settings.show_grid:
        ax.grid(True, linestyle="--", alpha=0.7)
    if settings.show_legend:
        # Move the legend outside the plot so it doesn't cover your data
        ax.legend(title="Datasets", bbox_to_anchor=(1.05, 1), loc='upper left')

    fig.tight_layout()
    return fig, ax