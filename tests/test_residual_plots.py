import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from odefit.data.dataset import Dataset
from odefit.plotting.residual_plots import (
    plot_residuals_over_time,
    save_residuals_plot,
)
from odefit.simulation.simulation_result import SimulationResult


def make_dataset():
    dataframe = pd.DataFrame(
        {
            "time": [0.0, 1.0, 2.0],
            "A": [1.0, 0.5, 0.25],
            "B": [0.0, 0.5, 0.75],
        }
    )

    return Dataset(
        raw_dataframe=dataframe,
        time_column="time",
        signal_columns=["A", "B"],
    )


def make_simulation_result():
    return SimulationResult(
        timepoints=np.array([0.0, 1.0, 2.0]),
        species=["A", "B"],
        values=np.array(
            [
                [1.0, 0.0],
                [0.45, 0.55],
                [0.20, 0.80],
            ]
        ),
    )


def test_plot_residuals_over_time():
    dataset = make_dataset()
    simulation_result = make_simulation_result()

    fig, ax = plot_residuals_over_time(
        dataset=dataset,
        simulation_result=simulation_result,
        species_mapping={
            "A": "A",
            "B": "B",
        },
    )

    assert fig is not None
    assert ax is not None

    # Two scatter collections for A/B residuals.
    assert len(ax.collections) == 2

    # Two zero-reference lines because currently one is added per mapped signal.
    assert len(ax.lines) == 2

    plt.close(fig)


def test_save_residuals_plot(tmp_path):
    dataset = make_dataset()
    simulation_result = make_simulation_result()

    output_path = tmp_path / "residuals.png"

    written_path = save_residuals_plot(
        dataset=dataset,
        simulation_result=simulation_result,
        species_mapping={
            "A": "A",
            "B": "B",
        },
        file_path=output_path,
    )

    assert written_path == output_path
    assert output_path.exists()
