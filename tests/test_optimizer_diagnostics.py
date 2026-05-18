import numpy as np

from odefit.fitting.fit_result import FitResult
from odefit.fitting.optimizer_diagnostics import (
    active_mask_to_string,
    build_optimizer_diagnostics_table,
    count_active_bounds,
    infer_optimizer_warning,
)
from odefit.simulation.simulation_result import SimulationResult


def make_simulation_result() -> SimulationResult:
    return SimulationResult(
        timepoints=np.array([0.0, 1.0]),
        species=["A"],
        values=np.array(
            [
                [1.0],
                [0.5],
            ]
        ),
    )


def make_fit_result(
    success: bool = True,
    status: int | None = 1,
    active_mask: np.ndarray | None = None,
) -> FitResult:
    return FitResult(
        success=success,
        message="Converged",
        fitted_parameters={"k1f": 0.5},
        initial_parameters={"k1f": 0.1},
        residuals=np.array([0.0, 0.1]),
        statistics={
            "rss": 0.01,
            "rmse": 0.05,
            "aic": -10.0,
            "bic": -9.0,
            "n_residuals": 2.0,
            "n_parameters": 1.0,
        },
        simulation_result=make_simulation_result(),
        nfev=5,
        cost=0.01,
        fitted_initial_conditions={"A": 1.0},
        initial_conditions={"A": 1.0},
        status=status,
        optimality=1e-8,
        active_mask=active_mask,
        njev=4,
    )


def test_count_active_bounds_none():
    assert count_active_bounds(None) == 0


def test_count_active_bounds_array():
    active_mask = np.array([0, -1, 0, 1])

    assert count_active_bounds(active_mask) == 2


def test_active_mask_to_string_none():
    assert active_mask_to_string(None) == ""


def test_active_mask_to_string_array():
    active_mask = np.array([0, -1, 1])

    assert active_mask_to_string(active_mask) == "0,-1,1"


def test_infer_optimizer_warning_none_for_successful_fit_without_active_bounds():
    fit_result = make_fit_result(
        success=True,
        status=1,
        active_mask=np.array([0, 0]),
    )

    assert infer_optimizer_warning(fit_result) == ""


def test_infer_optimizer_warning_failed_optimizer():
    fit_result = make_fit_result(
        success=False,
        status=-1,
        active_mask=np.array([0, 0]),
    )

    assert infer_optimizer_warning(fit_result) == "optimizer_failed"


def test_infer_optimizer_warning_max_evaluations():
    fit_result = make_fit_result(
        success=True,
        status=0,
        active_mask=np.array([0, 0]),
    )

    assert infer_optimizer_warning(fit_result) == "max_function_evaluations_reached"


def test_infer_optimizer_warning_active_bounds():
    fit_result = make_fit_result(
        success=True,
        status=1,
        active_mask=np.array([0, 1]),
    )

    assert infer_optimizer_warning(fit_result) == "one_or_more_variables_on_bounds"


def test_build_optimizer_diagnostics_table():
    fit_result = make_fit_result(
        success=True,
        status=1,
        active_mask=np.array([0, 1]),
    )

    table = build_optimizer_diagnostics_table(fit_result)

    assert list(table.columns) == ["diagnostic", "value"]

    diagnostics = dict(zip(table["diagnostic"], table["value"]))

    assert diagnostics["success"] is True
    assert diagnostics["status"] == 1
    assert diagnostics["nfev"] == 5
    assert diagnostics["njev"] == 4
    assert diagnostics["cost"] == 0.01
    assert diagnostics["optimality"] == 1e-8
    assert diagnostics["active_bound_count"] == 1
    assert diagnostics["active_mask"] == "0,1"
    assert diagnostics["warning"] == "one_or_more_variables_on_bounds"
