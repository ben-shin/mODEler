import numpy as np
import pandas as pd

from odefit.fitting.fit_result import FitResult


def count_active_bounds(active_mask: np.ndarray | None) -> int:
    """
    Count how many optimizer variables are on active bounds.

    In scipy.optimize.least_squares:
    - 0 means not active
    - -1 means active lower bound
    - 1 means active upper bound
    """

    if active_mask is None:
        return 0

    return int(np.count_nonzero(active_mask))


def active_mask_to_string(active_mask: np.ndarray | None) -> str:
    """
    Convert active_mask array into a compact string for CSV export.
    """

    if active_mask is None:
        return ""

    return ",".join(str(int(value)) for value in active_mask)


def infer_optimizer_warning(fit_result: FitResult) -> str:
    """
    Infer a compact optimizer-level warning.
    """

    if not fit_result.success:
        return "optimizer_failed"

    if fit_result.status == 0:
        return "max_function_evaluations_reached"

    if count_active_bounds(fit_result.active_mask) > 0:
        return "one_or_more_variables_on_bounds"

    return ""


def build_optimizer_diagnostics_table(
    fit_result: FitResult,
) -> pd.DataFrame:
    """
    Build optimizer-level diagnostics table.

    This table summarizes whether scipy.optimize.least_squares converged,
    why it stopped, and whether any variables were on active bounds.
    """

    active_bound_count = count_active_bounds(fit_result.active_mask)

    rows = [
        {
            "diagnostic": "success",
            "value": fit_result.success,
        },
        {
            "diagnostic": "status",
            "value": fit_result.status,
        },
        {
            "diagnostic": "message",
            "value": fit_result.message,
        },
        {
            "diagnostic": "nfev",
            "value": fit_result.nfev,
        },
        {
            "diagnostic": "njev",
            "value": fit_result.njev,
        },
        {
            "diagnostic": "cost",
            "value": fit_result.cost,
        },
        {
            "diagnostic": "optimality",
            "value": fit_result.optimality,
        },
        {
            "diagnostic": "active_bound_count",
            "value": active_bound_count,
        },
        {
            "diagnostic": "active_mask",
            "value": active_mask_to_string(fit_result.active_mask),
        },
        {
            "diagnostic": "warning",
            "value": infer_optimizer_warning(fit_result),
        },
    ]

    return pd.DataFrame(rows)
