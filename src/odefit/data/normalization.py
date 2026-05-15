import pandas as pd

from odefit.data.dataset import Dataset


def normalize_to_max(
    dataframe: pd.DataFrame,
    signal_columns: list[str],
) -> pd.DataFrame:
    """
    Normalize each signal column by its maximum value.
    """

    normalized = dataframe.copy()

    for column in signal_columns:
        max_value = normalized[column].max()

        if max_value == 0:
            raise ValueError(f"Cannot normalize column with maximum value 0: {column}")

        normalized[column] = normalized[column] / max_value

    return normalized


def min_max_normalize(
    dataframe: pd.DataFrame,
    signal_columns: list[str],
) -> pd.DataFrame:
    """
    Normalize each signal column to the range [0, 1].
    """

    normalized = dataframe.copy()

    for column in signal_columns:
        min_value = normalized[column].min()
        max_value = normalized[column].max()
        value_range = max_value - min_value

        if value_range == 0:
            raise ValueError(f"Cannot min-max normalize constant column: {column}")

        normalized[column] = (normalized[column] - min_value) / value_range

    return normalized


def divide_by_first_value(
    dataframe: pd.DataFrame,
    signal_columns: list[str],
) -> pd.DataFrame:
    """
    Divide each signal column by its first value.
    """

    normalized = dataframe.copy()

    for column in signal_columns:
        first_value = normalized[column].iloc[0]

        if first_value == 0:
            raise ValueError(f"Cannot divide by first value 0 in column: {column}")

        normalized[column] = normalized[column] / first_value

    return normalized


def subtract_baseline(
    dataframe: pd.DataFrame,
    signal_columns: list[str],
    baseline_points: int = 1,
) -> pd.DataFrame:
    """
    Subtract the mean of the first baseline_points from each signal column.
    """

    if baseline_points < 1:
        raise ValueError("baseline_points must be at least 1")

    if baseline_points > len(dataframe):
        raise ValueError("baseline_points cannot exceed number of rows")

    normalized = dataframe.copy()

    for column in signal_columns:
        baseline = normalized[column].iloc[:baseline_points].mean()
        normalized[column] = normalized[column] - baseline

    return normalized


def z_score_normalize(
    dataframe: pd.DataFrame,
    signal_columns: list[str],
) -> pd.DataFrame:
    """
    Z-score normalize each signal column.

    z = (x - mean) / standard deviation
    """

    normalized = dataframe.copy()

    for column in signal_columns:
        mean_value = normalized[column].mean()
        std_value = normalized[column].std()

        if std_value == 0:
            raise ValueError(f"Cannot z-score normalize constant column: {column}")

        normalized[column] = (normalized[column] - mean_value) / std_value

    return normalized


def apply_normalization(
    dataset: Dataset,
    method: str,
    baseline_points: int = 1,
) -> Dataset:
    """
    Apply a normalization method to a Dataset.

    Supported methods:
        none
        max
        min_max
        first
        baseline
        z_score
    """

    if method == "none":
        normalized_dataframe = dataset.raw_dataframe.copy()

    elif method == "max":
        normalized_dataframe = normalize_to_max(
            dataframe=dataset.raw_dataframe,
            signal_columns=dataset.signal_columns,
        )

    elif method == "min_max":
        normalized_dataframe = min_max_normalize(
            dataframe=dataset.raw_dataframe,
            signal_columns=dataset.signal_columns,
        )

    elif method == "first":
        normalized_dataframe = divide_by_first_value(
            dataframe=dataset.raw_dataframe,
            signal_columns=dataset.signal_columns,
        )

    elif method == "baseline":
        normalized_dataframe = subtract_baseline(
            dataframe=dataset.raw_dataframe,
            signal_columns=dataset.signal_columns,
            baseline_points=baseline_points,
        )

    elif method == "z_score":
        normalized_dataframe = z_score_normalize(
            dataframe=dataset.raw_dataframe,
            signal_columns=dataset.signal_columns,
        )

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return Dataset(
        raw_dataframe=dataset.raw_dataframe,
        normalized_dataframe=normalized_dataframe,
        time_column=dataset.time_column,
        signal_columns=dataset.signal_columns,
        normalization_method=method,
    )
