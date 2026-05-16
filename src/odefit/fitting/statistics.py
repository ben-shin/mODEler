import numpy as np


def calculate_rss(residuals: np.ndarray) -> float:
    """
    Residual sum of squares.
    """

    return float(np.sum(residuals**2))


def calculate_rmse(residuals: np.ndarray) -> float:
    """
    Root mean squared error.
    """

    return float(np.sqrt(np.mean(residuals**2)))


def calculate_aic(
    residuals: np.ndarray,
    number_of_parameters: int,
) -> float:
    """
    Akaike information criterion.

    Assumes Gaussian residuals.
    """

    n = len(residuals)
    rss = calculate_rss(residuals)

    if rss <= 0:
        rss = np.finfo(float).eps

    return float(n * np.log(rss / n) + 2 * number_of_parameters)


def calculate_bic(
    residuals: np.ndarray,
    number_of_parameters: int,
) -> float:
    """
    Bayesian information criterion.

    Assumes Gaussian residuals.
    """

    n = len(residuals)
    rss = calculate_rss(residuals)

    if rss <= 0:
        rss = np.finfo(float).eps

    return float(n * np.log(rss / n) + number_of_parameters * np.log(n))


def calculate_fit_statistics(
    residuals: np.ndarray,
    number_of_parameters: int,
) -> dict[str, float]:
    """
    Calculate common fit statistics.
    """

    return {
        "rss": calculate_rss(residuals),
        "rmse": calculate_rmse(residuals),
        "aic": calculate_aic(residuals, number_of_parameters),
        "bic": calculate_bic(residuals, number_of_parameters),
        "n_residuals": float(len(residuals)),
        "n_parameters": float(number_of_parameters),
    }
