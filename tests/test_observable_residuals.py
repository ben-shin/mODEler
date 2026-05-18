import numpy as np
import pandas as pd
import pytest

from odefit.data.dataset import Dataset
from odefit.fitting.observable_residuals import (
    calculate_observable_residuals,
    predict_observable_values,
)
from odefit.simulation.simulation_result import SimulationResult


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


def test_predict_observable_values():
    simulation_result = make_simulation_result()

    predictions = predict_observable_values(
        simulation_result=simulation_result,
        observable_parameters={
            "amide": {
                "species": "A",
                "scale": 2.0,
                "offset": 0.1,
            }
        },
    )

    assert predictions["amide"] == pytest.approx([2.1, 1.1, 0.6])


def test_calculate_observable_residuals():
    dataset = Dataset(
        raw_dataframe=pd.DataFrame(
            {
                "time": [0.0, 1.0, 2.0],
                "amide": [2.1, 1.0, 0.7],
            }
        ),
        time_column="time",
        signal_columns=["amide"],
    )

    simulation_result = make_simulation_result()

    residuals = calculate_observable_residuals(
        dataset=dataset,
        simulation_result=simulation_result,
        observable_parameters={
            "amide": {
                "species": "A",
                "scale": 2.0,
                "offset": 0.1,
            }
        },
    )

    assert residuals == pytest.approx([0.0, 0.1, -0.1])


def test_calculate_observable_residuals_normalized_data():
    raw_dataframe = pd.DataFrame(
        {
            "time": [0.0, 1.0],
            "amide": [100.0, 50.0],
        }
    )

    normalized_dataframe = pd.DataFrame(
        {
            "time": [0.0, 1.0],
            "amide": [1.0, 0.5],
        }
    )

    dataset = Dataset(
        raw_dataframe=raw_dataframe,
        normalized_dataframe=normalized_dataframe,
        normalization_method="manual",
        time_column="time",
        signal_columns=["amide"],
    )

    simulation_result = SimulationResult(
        timepoints=np.array([0.0, 1.0]),
        species=["A"],
        values=np.array(
            [
                [1.0],
                [0.4],
            ]
        ),
    )

    residuals = calculate_observable_residuals(
        dataset=dataset,
        simulation_result=simulation_result,
        observable_parameters={
            "amide": {
                "species": "A",
                "scale": 1.0,
                "offset": 0.0,
            }
        },
        use_normalized_data=True,
    )

    assert residuals == pytest.approx([0.0, -0.1])
