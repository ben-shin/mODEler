from dataclasses import dataclass


@dataclass
class PlotSettings:
    """
    Settings for plotting timecourses and fit diagnostics.
    """

    x_scale: str = "linear"
    y_scale: str = "linear"
    title: str | None = None
    x_label: str = "Time"
    y_label: str = "Signal"
    show_legend: bool = True
    show_grid: bool = True
