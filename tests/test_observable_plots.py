import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from odefit.data.dataset import Dataset
from odefit.plotting.observable_plots import (
    plot_observable_residuals_over_time,
    plot_observed_vs_predicted_observables,
    save_observable_residuals_plot,
    save_observed_vs_predicted_observables_plot,
)
from odefit.simulation.simulation_result import SimulationResult


def make_dataset():
    dataframe = pd.DataFrame(
        {
            "time": [0.0, 1.0, 2.0],
            "amide": [2.1, 1.1, 0.6],
        }
    )

    return Dataset(
        raw_dataframe=dataframe,
        time_column="time",
        signal_columns=["amide"],
    )


def make_simulation_result():
    return SimulationResult(
        timepoints=np.array([0.0, 1.0, 2.0]),
        species=["A"],
        values=np.array(
            [
                [1.0],
                [0.5],
                [0.25],
            ]
        ),
    )


def make_observable_parameters():
    return {
        "amide": {
            "species": "A",
            "scale": 2.0,
            "offset": 0.1,
        }
    }


def test_plot_observed_vs_predicted_observables():
    dataset = make_dataset()
    simulation_result = make_simulation_result()

    fig, ax = plot_observed_vs_predicted_observables(
        dataset=dataset,
        simulation_result=simulation_result,
        observable_parameters=make_observable_parameters(),
    )

    assert fig is not None
    assert ax is not None

    # One observed scatter collection, one fitted line.
    assert len(ax.collections) == 1
    assert len(ax.lines) == 1

    plt.close(fig)


def test_plot_observable_residuals_over_time():
    dataset = make_dataset()
    simulation_result = make_simulation_result()

    fig, ax = plot_observable_residuals_over_time(
        dataset=dataset,
        simulation_result=simulation_result,
        observable_parameters=make_observable_parameters(),
    )

    assert fig is not None
    assert ax is not None

    # One scatter collection for residuals.
    assert len(ax.collections) == 1

    # One zero-reference line.
    assert len(ax.lines) == 1

    plt.close(fig)


def test_save_observed_vs_predicted_observables_plot(tmp_path):
    dataset = make_dataset()
    simulation_result = make_simulation_result()

    output_path = tmp_path / "observable_observed_vs_fitted.png"

    written_path = save_observed_vs_predicted_observables_plot(
        dataset=dataset,
        simulation_result=simulation_result,
        observable_parameters=make_observable_parameters(),
        file_path=output_path,
    )

    assert written_path == output_path
    assert output_path.exists()


def test_save_observable_residuals_plot(tmp_path):
    dataset = make_dataset()
    simulation_result = make_simulation_result()

    output_path = tmp_path / "observable_residuals.png"

    written_path = save_observable_residuals_plot(
        dataset=dataset,
        simulation_result=simulation_result,
        observable_parameters=make_observable_parameters(),
        file_path=output_path,
    )

    assert written_path == output_path
    assert output_path.exists()
