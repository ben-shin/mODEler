import pandas as pd
import pytest

from odefit.data.dataset import Dataset
from odefit.data.normalization import (
    apply_normalization,
    divide_by_first_value,
    min_max_normalize,
    normalize_to_max,
    subtract_baseline,
    z_score_normalize,
)


def make_test_dataframe():
    return pd.DataFrame(
        {
            "time": [0.0, 1.0, 2.0],
            "A": [2.0, 4.0, 8.0],
            "B": [1.0, 3.0, 5.0],
        }
    )


def make_test_dataset():
    return Dataset(
        raw_dataframe=make_test_dataframe(),
        time_column="time",
        signal_columns=["A", "B"],
    )


def test_normalize_to_max():
    dataframe = make_test_dataframe()

    normalized = normalize_to_max(
        dataframe=dataframe,
        signal_columns=["A", "B"],
    )

    assert list(normalized["time"]) == [0.0, 1.0, 2.0]
    assert list(normalized["A"]) == [0.25, 0.5, 1.0]
    assert list(normalized["B"]) == [0.2, 0.6, 1.0]


def test_min_max_normalize():
    dataframe = make_test_dataframe()

    normalized = min_max_normalize(
        dataframe=dataframe,
        signal_columns=["A", "B"],
    )

    assert list(normalized["time"]) == [0.0, 1.0, 2.0]
    assert list(normalized["A"]) == [0.0, pytest.approx(1.0 / 3.0), 1.0]
    assert list(normalized["B"]) == [0.0, 0.5, 1.0]


def test_divide_by_first_value():
    dataframe = make_test_dataframe()

    normalized = divide_by_first_value(
        dataframe=dataframe,
        signal_columns=["A", "B"],
    )

    assert list(normalized["A"]) == [1.0, 2.0, 4.0]
    assert list(normalized["B"]) == [1.0, 3.0, 5.0]


def test_subtract_baseline_one_point():
    dataframe = make_test_dataframe()

    normalized = subtract_baseline(
        dataframe=dataframe,
        signal_columns=["A", "B"],
        baseline_points=1,
    )

    assert list(normalized["A"]) == [0.0, 2.0, 6.0]
    assert list(normalized["B"]) == [0.0, 2.0, 4.0]


def test_subtract_baseline_two_points():
    dataframe = make_test_dataframe()

    normalized = subtract_baseline(
        dataframe=dataframe,
        signal_columns=["A", "B"],
        baseline_points=2,
    )

    assert list(normalized["A"]) == [-1.0, 1.0, 5.0]
    assert list(normalized["B"]) == [-1.0, 1.0, 3.0]


def test_z_score_normalize():
    dataframe = make_test_dataframe()

    normalized = z_score_normalize(
        dataframe=dataframe,
        signal_columns=["A"],
    )

    assert normalized["A"].mean() == pytest.approx(0.0)
    assert normalized["A"].std() == pytest.approx(1.0)


def test_apply_normalization_max():
    dataset = make_test_dataset()

    normalized_dataset = apply_normalization(
        dataset=dataset,
        method="max",
    )

    assert normalized_dataset.normalized_dataframe is not None
    assert normalized_dataset.normalization_method == "max"
    assert list(normalized_dataset.get_signal_values("A", normalized=True)) == [
        0.25,
        0.5,
        1.0,
    ]


def test_apply_normalization_none():
    dataset = make_test_dataset()

    normalized_dataset = apply_normalization(
        dataset=dataset,
        method="none",
    )

    assert normalized_dataset.normalized_dataframe is not None
    assert list(normalized_dataset.get_signal_values("A", normalized=True)) == [
        2.0,
        4.0,
        8.0,
    ]


def test_unknown_normalization_method_raises_error():
    dataset = make_test_dataset()

    with pytest.raises(ValueError):
        apply_normalization(
            dataset=dataset,
            method="unknown",
        )


def test_normalize_to_max_zero_column_raises_error():
    dataframe = pd.DataFrame(
        {
            "time": [0.0, 1.0],
            "A": [0.0, 0.0],
        }
    )

    with pytest.raises(ValueError):
        normalize_to_max(
            dataframe=dataframe,
            signal_columns=["A"],
        )


def test_min_max_constant_column_raises_error():
    dataframe = pd.DataFrame(
        {
            "time": [0.0, 1.0],
            "A": [5.0, 5.0],
        }
    )

    with pytest.raises(ValueError):
        min_max_normalize(
            dataframe=dataframe,
            signal_columns=["A"],
        )


def test_divide_by_first_value_zero_raises_error():
    dataframe = pd.DataFrame(
        {
            "time": [0.0, 1.0],
            "A": [0.0, 5.0],
        }
    )

    with pytest.raises(ValueError):
        divide_by_first_value(
            dataframe=dataframe,
            signal_columns=["A"],
        )


def test_z_score_constant_column_raises_error():
    dataframe = pd.DataFrame(
        {
            "time": [0.0, 1.0],
            "A": [5.0, 5.0],
        }
    )

    with pytest.raises(ValueError):
        z_score_normalize(
            dataframe=dataframe,
            signal_columns=["A"],
        )


def test_get_normalized_signal_without_normalized_dataframe_raises_error():
    dataset = make_test_dataset()

    with pytest.raises(ValueError):
        dataset.get_signal_values("A", normalized=True)
