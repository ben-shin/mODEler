import pandas as pd
import pytest

from odefit.data.csv_reader import read_csv_dataset
from odefit.data.validation import validate_dataframe


def test_read_csv_dataset(tmp_path):
    csv_path = tmp_path / "data.csv"

    dataframe = pd.DataFrame(
        {
            "time": [0.0, 1.0, 2.0],
            "A": [1.0, 0.8, 0.6],
            "B": [0.0, 0.1, 0.2],
        }
    )

    dataframe.to_csv(csv_path, index=False)

    dataset = read_csv_dataset(
        file_path=str(csv_path),
        time_column="time",
        signal_columns=["A", "B"],
    )

    assert dataset.time_column == "time"
    assert dataset.signal_columns == ["A", "B"]
    assert list(dataset.time_values) == [0.0, 1.0, 2.0]
    assert list(dataset.get_signal_values("A")) == [1.0, 0.8, 0.6]


def test_missing_time_column_raises_error():
    dataframe = pd.DataFrame(
        {
            "t": [0.0, 1.0],
            "A": [1.0, 0.5],
        }
    )

    with pytest.raises(ValueError):
        validate_dataframe(
            dataframe=dataframe,
            time_column="time",
            signal_columns=["A"],
        )


def test_missing_signal_column_raises_error():
    dataframe = pd.DataFrame(
        {
            "time": [0.0, 1.0],
            "A": [1.0, 0.5],
        }
    )

    with pytest.raises(ValueError):
        validate_dataframe(
            dataframe=dataframe,
            time_column="time",
            signal_columns=["B"],
        )


def test_non_numeric_time_column_raises_error():
    dataframe = pd.DataFrame(
        {
            "time": ["zero", "one"],
            "A": [1.0, 0.5],
        }
    )

    with pytest.raises(ValueError):
        validate_dataframe(
            dataframe=dataframe,
            time_column="time",
            signal_columns=["A"],
        )


def test_time_values_must_be_increasing():
    dataframe = pd.DataFrame(
        {
            "time": [0.0, 2.0, 1.0],
            "A": [1.0, 0.5, 0.2],
        }
    )

    with pytest.raises(ValueError):
        validate_dataframe(
            dataframe=dataframe,
            time_column="time",
            signal_columns=["A"],
        )
