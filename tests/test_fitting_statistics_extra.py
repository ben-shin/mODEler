import numpy as np
import pytest

from odefit.fitting.statistics import (
    calculate_aic,
    calculate_bic,
    calculate_fit_statistics,
    calculate_rmse,
    calculate_rss,
)


def test_calculate_rss():
    residuals = np.array([1.0, -2.0, 3.0])

    assert calculate_rss(residuals) == 14.0


def test_calculate_rmse():
    residuals = np.array([1.0, -2.0, 3.0])

    assert calculate_rmse(residuals) == np.sqrt(14.0 / 3.0)


def test_aic_and_bic_are_finite_for_zero_residuals():
    residuals = np.array([0.0, 0.0, 0.0])

    aic = calculate_aic(
        residuals=residuals,
        number_of_parameters=1,
    )

    bic = calculate_bic(
        residuals=residuals,
        number_of_parameters=1,
    )

    assert np.isfinite(aic)
    assert np.isfinite(bic)


def test_calculate_fit_statistics_contains_expected_keys():
    residuals = np.array([1.0, -2.0, 3.0])

    statistics = calculate_fit_statistics(
        residuals=residuals,
        number_of_parameters=2,
    )

    assert set(statistics.keys()) == {
        "rss",
        "rmse",
        "aic",
        "bic",
        "n_residuals",
        "n_parameters",
    }

    assert statistics["rss"] == 14.0
    assert statistics["n_residuals"] == 3.0
    assert statistics["n_parameters"] == 2.0
