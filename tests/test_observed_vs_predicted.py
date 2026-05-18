import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from odefit.data.dataset import Dataset
from odefit.plotting.observed_vs_predicted import (
    plot_observed_vs_fitted_timecourse,
    save_observed_vs_fitted_plot,
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


def test_plot_observed_vs_fitted_timecourse():
    dataset = make_dataset()
    simulation_result = make_simulation_result()

    fig, ax = plot_observed_vs_fitted_timecourse(
        dataset=dataset,
        simulation_result=simulation_result,
        species_mapping={
            "A": "A",
            "B": "B",
        },
    )

    assert fig is not None
    assert ax is not None

    # Two observed scatter collections, two fitted lines.
    assert len(ax.collections) == 2
    assert len(ax.lines) == 2

    plt.close(fig)


def test_save_observed_vs_fitted_plot(tmp_path):
    dataset = make_dataset()
    simulation_result = make_simulation_result()

    output_path = tmp_path / "observed_vs_fitted.png"

    written_path = save_observed_vs_fitted_plot(
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
