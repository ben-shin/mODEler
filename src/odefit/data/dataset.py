from dataclasses import dataclass

import pandas as pd


@dataclass
class Dataset:
    """
    Container for imported experimental data.
    """

    raw_dataframe: pd.DataFrame
    time_column: str
    signal_columns: list[str]

    @property
    def time_values(self):
        return self.raw_dataframe[self.time_column].to_numpy()

    def get_signal_values(self, signal_column: str):
        return self.raw_dataframe[signal_column].to_numpy()
