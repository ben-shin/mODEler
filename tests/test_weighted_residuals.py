import numpy as np
import pandas as pd
import pytest

from odefit.data.dataset import Dataset
from odefit.fitting.observable_residuals import calculate_observable_residuals
from odefit.fitting.residuals import calculate_residuals, get_signal_weight
from odefit.simulation.simulation_result import SimulationResult


def make_dataset():
    return Dataset(
        raw_dataframe=pd.DataFrame(
            {
                "time": [0.0, 1.0],
                "A": [1.0, 0.5],
                "amide": [2.0, 1.0],
            }
        ),
        time_column="time",
        signal_columns=["A", "amide"],
    )


def test_get_signal_weight_defaults_to_one():
    assert get_signal_weight("A") == 1.0


def test_get_signal_weight_uses_provided_weight():
    weight = get_signal_weight(
        data_column="A",
        signal_weights={"A": 2.0},
    )

    assert weight == 2.0


def test_get_signal_weight_uses_one_for_missing_column():
    weight = get_signal_weight(
        data_column="B",
        signal_weights={"A": 2.0},
    )

    assert weight == 1.0


def test_calculate_residuals_with_signal_weights():
    dataset = make_dataset()

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
        signal_weights={"A": 2.0},
    )

    # Unweighted residuals:
    # [1.0 - 1.0, 0.4 - 0.5] = [0.0, -0.1]
    #
    # Weighted by 2:
    # [0.0, -0.2]
    assert residuals == pytest.approx([0.0, -0.2])


def test_calculate_observable_residuals_with_signal_weights():
    dataset = make_dataset()

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
                "scale": 2.0,
                "offset": 0.0,
            }
        },
        signal_weights={"amide": 3.0},
    )

    # Predicted amide:
    # [2.0, 0.8]
    #
    # Observed amide:
    # [2.0, 1.0]
    #
    # Unweighted residual:
    # [0.0, -0.2]
    #
    # Weighted by 3:
    # [0.0, -0.6]
    assert residuals == pytest.approx([0.0, -0.6])
