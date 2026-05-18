import numpy as np
import pandas as pd
import pytest

from odefit.fitting.fit_result import FitResult
from odefit.fitting.model_comparison import (
    build_model_comparison_table,
    build_ranked_model_comparison_table,
    calculate_delta_metric,
    get_best_model_name,
)
from odefit.simulation.simulation_result import SimulationResult


def make_simulation_result() -> SimulationResult:
    return SimulationResult(
        timepoints=np.array([0.0, 1.0]),
        species=["A", "B"],
        values=np.array(
            [
                [1.0, 0.0],
                [0.5, 0.5],
            ]
        ),
    )


def make_fit_result(
    rss: float,
    rmse: float,
    aic: float,
    bic: float,
    cost: float,
    n_parameters: float,
    success: bool = True,
    message: str = "Converged",
) -> FitResult:
    return FitResult(
        success=success,
        message=message,
        fitted_parameters={"k1f": 0.5},
        initial_parameters={"k1f": 0.1},
        residuals=np.array([0.0, 0.1]),
        statistics={
            "rss": rss,
            "rmse": rmse,
            "aic": aic,
            "bic": bic,
            "n_residuals": 2.0,
            "n_parameters": n_parameters,
        },
        simulation_result=make_simulation_result(),
        nfev=5,
        cost=cost,
        fitted_initial_conditions={"A": 1.0, "B": 0.0},
        initial_conditions={"A": 1.0, "B": 0.0},
    )


def test_build_model_comparison_table_sorts_by_aic():
    fit_results = {
        "first_order": make_fit_result(
            rss=1.0,
            rmse=0.5,
            aic=10.0,
            bic=11.0,
            cost=0.5,
            n_parameters=1.0,
        ),
        "dimerization": make_fit_result(
            rss=0.5,
            rmse=0.25,
            aic=5.0,
            bic=6.0,
            cost=0.25,
            n_parameters=2.0,
        ),
    }

    table = build_model_comparison_table(
        fit_results=fit_results,
        sort_by="aic",
    )

    assert isinstance(table, pd.DataFrame)

    assert list(table.columns) == [
        "rank",
        "model",
        "success",
        "rss",
        "rmse",
        "aic",
        "bic",
        "n_residuals",
        "n_parameters",
        "cost",
        "nfev",
        "message",
    ]

    assert list(table["rank"]) == [1, 2]
    assert list(table["model"]) == ["dimerization", "first_order"]
    assert list(table["aic"]) == [5.0, 10.0]


def test_build_model_comparison_table_sorts_by_bic():
    fit_results = {
        "model_a": make_fit_result(
            rss=1.0,
            rmse=0.5,
            aic=5.0,
            bic=20.0,
            cost=0.5,
            n_parameters=1.0,
        ),
        "model_b": make_fit_result(
            rss=0.8,
            rmse=0.4,
            aic=10.0,
            bic=15.0,
            cost=0.4,
            n_parameters=2.0,
        ),
    }

    table = build_model_comparison_table(
        fit_results=fit_results,
        sort_by="bic",
    )

    assert list(table["model"]) == ["model_b", "model_a"]
    assert list(table["bic"]) == [15.0, 20.0]


def test_get_best_model_name():
    fit_results = {
        "first_order": make_fit_result(
            rss=1.0,
            rmse=0.5,
            aic=10.0,
            bic=11.0,
            cost=0.5,
            n_parameters=1.0,
        ),
        "dimerization": make_fit_result(
            rss=0.5,
            rmse=0.25,
            aic=5.0,
            bic=6.0,
            cost=0.25,
            n_parameters=2.0,
        ),
    }

    best_model = get_best_model_name(
        fit_results=fit_results,
        sort_by="aic",
    )

    assert best_model == "dimerization"


def test_calculate_delta_metric():
    table = pd.DataFrame(
        {
            "model": ["model_a", "model_b"],
            "aic": [5.0, 8.0],
        }
    )

    table_with_delta = calculate_delta_metric(
        comparison_table=table,
        metric="aic",
    )

    assert list(table_with_delta["delta_aic"]) == [0.0, 3.0]


def test_build_ranked_model_comparison_table_adds_delta_columns():
    fit_results = {
        "first_order": make_fit_result(
            rss=1.0,
            rmse=0.5,
            aic=10.0,
            bic=11.0,
            cost=0.5,
            n_parameters=1.0,
        ),
        "dimerization": make_fit_result(
            rss=0.5,
            rmse=0.25,
            aic=5.0,
            bic=7.0,
            cost=0.25,
            n_parameters=2.0,
        ),
    }

    table = build_ranked_model_comparison_table(
        fit_results=fit_results,
        sort_by="aic",
    )

    assert "delta_aic" in table.columns
    assert "delta_bic" in table.columns

    assert list(table["model"]) == ["dimerization", "first_order"]
    assert list(table["delta_aic"]) == [0.0, 5.0]
    assert list(table["delta_bic"]) == [0.0, 4.0]


def test_empty_fit_results_raises_error():
    with pytest.raises(ValueError):
        build_model_comparison_table({})


def test_invalid_sort_column_raises_error():
    fit_results = {
        "model_a": make_fit_result(
            rss=1.0,
            rmse=0.5,
            aic=10.0,
            bic=11.0,
            cost=0.5,
            n_parameters=1.0,
        )
    }

    with pytest.raises(ValueError):
        build_model_comparison_table(
            fit_results=fit_results,
            sort_by="not_a_column",
        )


def test_sort_by_missing_values_raises_error():
    fit_result = make_fit_result(
        rss=1.0,
        rmse=0.5,
        aic=10.0,
        bic=11.0,
        cost=0.5,
        n_parameters=1.0,
    )

    fit_result.statistics["aic"] = None

    with pytest.raises(ValueError):
        build_model_comparison_table(
            fit_results={"model_a": fit_result},
            sort_by="aic",
        )


def test_calculate_delta_metric_missing_metric_raises_error():
    table = pd.DataFrame(
        {
            "model": ["model_a"],
            "aic": [5.0],
        }
    )

    with pytest.raises(ValueError):
        calculate_delta_metric(
            comparison_table=table,
            metric="bic",
        )


def test_calculate_delta_metric_empty_table_raises_error():
    table = pd.DataFrame(
        {
            "model": [],
            "aic": [],
        }
    )

    with pytest.raises(ValueError):
        calculate_delta_metric(
            comparison_table=table,
            metric="aic",
        )
