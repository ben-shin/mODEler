import numpy as np
import pandas as pd
import pytest

from odefit.data.dataset import Dataset
from odefit.fitting.observable_residual_table import build_observable_residual_table
from odefit.simulation.simulation_result import SimulationResult


def test_build_observable_residual_table():
    dataset = Dataset(
        raw_dataframe=pd.DataFrame(
            {
                "time": [0.0, 1.0, 2.0],
                "peak_A23": [2.1, 1.1, 0.6],
                "peak_G45": [3.2, 1.7, 0.95],
            }
        ),
        time_column="time",
        signal_columns=["peak_A23", "peak_G45"],
    )

    simulation_result = SimulationResult(
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

    table = build_observable_residual_table(
        dataset=dataset,
        simulation_result=simulation_result,
        observable_parameters={
            "peak_A23": {
                "species": "A",
                "scale": 2.0,
                "offset": 0.1,
            },
            "peak_G45": {
                "species": "A",
                "scale": 3.0,
                "offset": 0.2,
            },
        },
    )

    assert list(table.columns) == [
        "time",
        "peak_A23_observed",
        "peak_A23_fit",
        "peak_A23_residual",
        "peak_G45_observed",
        "peak_G45_fit",
        "peak_G45_residual",
    ]

    assert list(table["peak_A23_fit"]) == pytest.approx([2.1, 1.1, 0.6])
    assert list(table["peak_A23_residual"]) == pytest.approx([0.0, 0.0, 0.0])

    assert list(table["peak_G45_fit"]) == pytest.approx([3.2, 1.7, 0.95])
    assert list(table["peak_G45_residual"]) == pytest.approx([0.0, 0.0, 0.0])


def test_build_observable_residual_table_length_mismatch_raises_error():
    dataset = Dataset(
        raw_dataframe=pd.DataFrame(
            {
                "time": [0.0, 1.0, 2.0],
                "peak_A23": [2.1, 1.1, 0.6],
            }
        ),
        time_column="time",
        signal_columns=["peak_A23"],
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
        build_observable_residual_table(
            dataset=dataset,
            simulation_result=simulation_result,
            observable_parameters={
                "peak_A23": {
                    "species": "A",
                    "scale": 2.0,
                    "offset": 0.1,
                }
            },
        )
