import pandas as pd

from odefit.fitting.fit_result import FitResult


def build_model_comparison_table(
    fit_results: dict[str, FitResult],
    sort_by: str = "aic",
) -> pd.DataFrame:
    """
    Build a model comparison table from multiple FitResult objects.

    fit_results maps model names to FitResult objects.

    Example:
        {
            "first_order": fit_result_1,
            "dimerization": fit_result_2,
        }

    The table is sorted by sort_by, where lower values are usually better
    for RSS, RMSE, AIC, BIC, and cost.
    """

    if not fit_results:
        raise ValueError("At least one FitResult is required")

    rows = []

    for model_name, fit_result in fit_results.items():
        statistics = fit_result.statistics

        rows.append(
            {
                "model": model_name,
                "success": fit_result.success,
                "message": fit_result.message,
                "rss": statistics.get("rss"),
                "rmse": statistics.get("rmse"),
                "aic": statistics.get("aic"),
                "bic": statistics.get("bic"),
                "n_residuals": statistics.get("n_residuals"),
                "n_parameters": statistics.get("n_parameters"),
                "cost": fit_result.cost,
                "nfev": fit_result.nfev,
            }
        )

    table = pd.DataFrame(rows)

    if sort_by not in table.columns:
        raise ValueError(f"Cannot sort model comparison table by: {sort_by}")

    if table[sort_by].isna().any():
        raise ValueError(f"Cannot sort by column with missing values: {sort_by}")

    table = table.sort_values(sort_by).reset_index(drop=True)

    table["rank"] = range(1, len(table) + 1)

    ordered_columns = [
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

    return table[ordered_columns]


def get_best_model_name(
    fit_results: dict[str, FitResult],
    sort_by: str = "aic",
) -> str:
    """
    Return the best model name according to the chosen sorting metric.
    """

    table = build_model_comparison_table(
        fit_results=fit_results,
        sort_by=sort_by,
    )

    return str(table.iloc[0]["model"])


def calculate_delta_metric(
    comparison_table: pd.DataFrame,
    metric: str,
    output_column: str | None = None,
) -> pd.DataFrame:
    """
    Add a delta column relative to the best model.

    Example:
        delta_aic = AIC - best_AIC
        delta_bic = BIC - best_BIC
    """

    if metric not in comparison_table.columns:
        raise ValueError(f"Metric not found in comparison table: {metric}")

    if comparison_table.empty:
        raise ValueError("Comparison table is empty")

    if output_column is None:
        output_column = f"delta_{metric}"

    table = comparison_table.copy()

    best_value = table[metric].min()

    table[output_column] = table[metric] - best_value

    return table


def build_ranked_model_comparison_table(
    fit_results: dict[str, FitResult],
    sort_by: str = "aic",
) -> pd.DataFrame:
    """
    Build a model comparison table with delta AIC and delta BIC columns.
    """

    table = build_model_comparison_table(
        fit_results=fit_results,
        sort_by=sort_by,
    )

    if "aic" in table.columns:
        table = calculate_delta_metric(
            comparison_table=table,
            metric="aic",
            output_column="delta_aic",
        )

    if "bic" in table.columns:
        table = calculate_delta_metric(
            comparison_table=table,
            metric="bic",
            output_column="delta_bic",
        )

    return table
