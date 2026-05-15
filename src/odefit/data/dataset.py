from dataclasses import dataclass

import pandas as pd


@dataclass
class Dataset:
    """
    Container for imported experimental data.

    raw_dataframe stores the original imported data
    normalized_dataframe stores normalized signal data if normalization is applied
    """

    raw_dataframe: pd.DataFrame
    time_column: str
    signal_columns: list[str]
    normalized_dataframe: pd.DataFrame | None = None
    normalization_method: str | None = None

    @property
    def time_values(self):
        return self.raw_dataframe[self.time_column].to_numpy()

    def get_signal_values(
        self,
        signal_column: str,
        normalized: bool = False,
    ):
        """
        Return values for one signal column.

        If normalized=True, return values from normalized_dataframe.
        Otherwise, return values from raw_dataframe.
        """

        if normalized:
            if self.normalized_dataframe is None:
                raise ValueError("Normalized data is not available")

            return self.normalized_dataframe[signal_column].to_numpy()

        return self.raw_dataframe[signal_column].to_numpy()
