import pandas as pd


def validate_dataframe(
    dataframe: pd.DataFrame,
    time_column: str,
    signal_columns: list[str],
) -> None:
    """
    Validate imported data before creating a Dataset.
    """

    if dataframe.empty:
        raise ValueError("Dataframe is empty")

    if time_column not in dataframe.columns:
        raise ValueError(f"Time column not found: {time_column}")

    for signal_column in signal_columns:
        if signal_column not in dataframe.columns:
            raise ValueError(f"Signal column not found: {signal_column}")

    if len(signal_columns) == 0:
        raise ValueError("At least one signal column is required")

    if dataframe[time_column].isna().any():
        raise ValueError("Time column contains missing values")

    if not pd.api.types.is_numeric_dtype(dataframe[time_column]):
        raise ValueError("Time column must be numeric")

    for signal_column in signal_columns:
        if dataframe[signal_column].isna().any():
            raise ValueError(f"Signal column contains missing values: {signal_column}")

        if not pd.api.types.is_numeric_dtype(dataframe[signal_column]):
            raise ValueError(f"Signal column must be numeric: {signal_column}")

    time_values = dataframe[time_column].to_numpy()

    if not all(
        time_values[i] < time_values[i + 1] for i in range(len(time_values) - 1)
    ):
        raise ValueError("Time values must be strictly increasing")
