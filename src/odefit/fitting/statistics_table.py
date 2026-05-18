import pandas as pd

from odefit.fitting.fit_result import FitResult


def build_statistics_table(
    fit_result: FitResult,
) -> pd.DataFrame:
    """
    Build a tidy statistics table for GUI display and CSV export.
    """

    rows = []

    rows.append({"statistic": "success", "value": fit_result.success})
    rows.append({"statistic": "message", "value": fit_result.message})
    rows.append({"statistic": "nfev", "value": fit_result.nfev})
    rows.append({"statistic": "cost", "value": fit_result.cost})

    for statistic_name, statistic_value in fit_result.statistics.items():
        rows.append(
            {
                "statistic": statistic_name,
                "value": statistic_value,
            }
        )

    return pd.DataFrame(rows)
