import numpy as np
import pandas as pd
import pytest

from odefit.data.dataset import Dataset
from odefit.fitting.residual_table import build_residual_table
from odefit.simulation.simulation_result import SimulationResult


def test_build_residual_table():
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

    table = build_residual_table(
        dataset=dataset,
        simulation_result=simulation_result,
        species_mapping={
            "A": "A",
            "B": "B",
        },
    )

    assert list(table.columns) == [
        "time",
        "A_observed",
        "A_fit",
        "A_residual",
        "B_observed",
        "B_fit",
        "B_residual",
    ]

    assert list(table["A_residual"]) == pytest.approx([0.0, -0.1])
    assert list(table["B_residual"]) == pytest.approx([0.0, 0.1])
