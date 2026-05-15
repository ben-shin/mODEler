import pandas as pd

from odefit.data.dataset import Dataset
from odefit.data.validation import validate_dataframe


def read_csv_dataset(
    file_path: str,
    time_column: str,
    signal_columns: list[str],
) -> Dataset:
    """
    Read a CSV file and return a validated Dataset.
    """

    dataframe = pd.read_csv(file_path)

    validate_dataframe(
        dataframe=dataframe,
        time_column=time_column,
        signal_columns=signal_columns,
    )

    return Dataset(
        raw_dataframe=dataframe,
        time_column=time_column,
        signal_columns=signal_columns,
    )
