import numpy as np
import pandas as pd
import pytest

from odefit.data.dataset import Dataset
from odefit.fitting.residuals import calculate_residuals
from odefit.simulation.simulation_result import SimulationResult


def test_calculate_residuals_raw_data():
    dataset = Dataset(
        raw_dataframe=pd.DataFrame(
            {
                "time": [0.0, 1.0],
                "A": [1.0, 0.5],
                "B": [0.0, 0.5],
            }
        ),
        time_column="time",
        signal_columns=["A", "B"],
    )

    simulation_result = SimulationResult(
        timepoints=np.array([0.0, 1.0]),
        species=["A", "B"],
        values=np.array(
            [
                [1.0, 0.0],
                [0.4, 0.6],
            ]
        ),
    )

    residuals = calculate_residuals(
        dataset=dataset,
        simulation_result=simulation_result,
        species_mapping={
            "A": "A",
            "B": "B",
        },
    )

    assert residuals == pytest.approx([0.0, -0.1, 0.0, 0.1])


def test_calculate_residuals_normalized_data():
    raw_dataframe = pd.DataFrame(
        {
            "time": [0.0, 1.0],
            "A": [10.0, 5.0],
        }
    )

    normalized_dataframe = pd.DataFrame(
        {
            "time": [0.0, 1.0],
            "A": [1.0, 0.5],
        }
    )

    dataset = Dataset(
        raw_dataframe=raw_dataframe,
        normalized_dataframe=normalized_dataframe,
        normalization_method="max",
        time_column="time",
        signal_columns=["A"],
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

    residuals = calculate_residuals(
        dataset=dataset,
        simulation_result=simulation_result,
        species_mapping={"A": "A"},
        use_normalized_data=True,
    )

    assert residuals == pytest.approx([0.0, -0.1])


def test_calculate_residuals_length_mismatch_raises_error():
    dataset = Dataset(
        raw_dataframe=pd.DataFrame(
            {
                "time": [0.0, 1.0, 2.0],
                "A": [1.0, 0.5, 0.25],
            }
        ),
        time_column="time",
        signal_columns=["A"],
    )

    simulation_result = SimulationResult(
        timepoints=np.array([0.0, 1.0]),
        species=["A"],
        values=np.array(
            [
                [1.0],
                [0.5],
            ]
        ),
    )

    with pytest.raises(ValueError):
        calculate_residuals(
            dataset=dataset,
            simulation_result=simulation_result,
            species_mapping={"A": "A"},
        )


def test_calculate_residuals_missing_model_species_raises_error():
    dataset = Dataset(
        raw_dataframe=pd.DataFrame(
            {
                "time": [0.0, 1.0],
                "A": [1.0, 0.5],
            }
        ),
        time_column="time",
        signal_columns=["A"],
    )

    simulation_result = SimulationResult(
        timepoints=np.array([0.0, 1.0]),
        species=["B"],
        values=np.array(
            [
                [0.0],
                [0.5],
            ]
        ),
    )

    with pytest.raises(ValueError):
        calculate_residuals(
            dataset=dataset,
            simulation_result=simulation_result,
            species_mapping={"A": "A"},
        )
