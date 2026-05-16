import numpy as np
import pandas as pd

from odefit.fitting.fit_result import FitResult
from odefit.fitting.statistics_table import build_statistics_table
from odefit.simulation.simulation_result import SimulationResult


def test_build_statistics_table():
    simulation_result = SimulationResult(
        timepoints=np.array([0.0, 1.0]),
        species=["A", "B"],
        values=np.array(
            [
                [1.0, 0.0],
                [0.5, 0.5],
            ]
        ),
    )

    fit_result = FitResult(
        success=True,
        message="Converged",
        fitted_parameters={"k1f": 0.5},
        initial_parameters={"k1f": 0.1},
        residuals=np.array([0.0, 0.1]),
        statistics={
            "rss": 0.01,
            "rmse": 0.05,
            "aic": -10.0,
            "bic": -9.0,
        },
        simulation_result=simulation_result,
        nfev=8,
        cost=0.005,
    )

    table = build_statistics_table(fit_result)

    assert isinstance(table, pd.DataFrame)
    assert list(table.columns) == ["statistic", "value"]

    statistics = dict(zip(table["statistic"], table["value"]))

    assert statistics["success"] is True
    assert statistics["message"] == "Converged"
    assert statistics["nfev"] == 8
    assert statistics["cost"] == 0.005
    assert statistics["rss"] == 0.01
    assert statistics["rmse"] == 0.05
    assert statistics["aic"] == -10.0
    assert statistics["bic"] == -9.0
