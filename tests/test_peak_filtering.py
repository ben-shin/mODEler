import numpy as np
import pandas as pd
import pytest

from odefit.data.peak_filtering import (
    build_peak_filtering_table,
    calculate_dynamic_range,
    calculate_initial_intensity,
    calculate_missing_fraction,
    filter_peak_columns,
    infer_numeric_signal_columns,
    interpolate_missing_values,
    prepare_peak_dataframe,
)


def test_infer_numeric_signal_columns_excludes_time_and_metadata():
    dataframe = pd.DataFrame(
        {
            "time": [0.0, 1.0],
            "assignment": ["A23", "G45"],
            "A23_HN": [1000.0, 900.0],
            "G45_HN": [800.0, 750.0],
        }
    )

    signal_columns = infer_numeric_signal_columns(
        dataframe=dataframe,
        time_column="time",
        exclude_columns=["assignment"],
    )

    assert signal_columns == ["A23_HN", "G45_HN"]


def test_infer_numeric_signal_columns_rejects_missing_time_column():
    dataframe = pd.DataFrame(
        {
            "A23_HN": [1000.0, 900.0],
        }
    )

    with pytest.raises(ValueError):
        infer_numeric_signal_columns(
            dataframe=dataframe,
            time_column="time",
        )


def test_calculate_missing_fraction():
    dataframe = pd.DataFrame(
        {
            "A23_HN": [1.0, np.nan, 3.0, np.nan],
        }
    )

    fraction = calculate_missing_fraction(
        dataframe=dataframe,
        column="A23_HN",
    )

    assert fraction == 0.5


def test_calculate_initial_intensity():
    dataframe = pd.DataFrame(
        {
            "A23_HN": [1000.0, 900.0, 800.0],
        }
    )

    intensity = calculate_initial_intensity(
        dataframe=dataframe,
        column="A23_HN",
        n_initial_points=2,
    )

    assert intensity == 950.0


def test_calculate_dynamic_range():
    dataframe = pd.DataFrame(
        {
            "A23_HN": [1000.0, 900.0, 850.0],
        }
    )

    dynamic_range = calculate_dynamic_range(
        dataframe=dataframe,
        column="A23_HN",
    )

    assert dynamic_range == 150.0


def test_filter_peak_columns_removes_too_many_missing_values():
    dataframe = pd.DataFrame(
        {
            "time": [0.0, 1.0, 2.0, 3.0],
            "good_peak": [1.0, 0.9, 0.8, 0.7],
            "bad_peak": [1.0, np.nan, np.nan, 0.7],
        }
    )

    result = filter_peak_columns(
        dataframe=dataframe,
        signal_columns=["good_peak", "bad_peak"],
        max_missing_fraction=0.25,
    )

    assert result.kept_columns == ["good_peak"]
    assert result.removed_columns == ["bad_peak"]
    assert "missing fraction" in result.removal_reasons["bad_peak"]


def test_filter_peak_columns_removes_low_initial_intensity():
    dataframe = pd.DataFrame(
        {
            "time": [0.0, 1.0, 2.0],
            "strong_peak": [1000.0, 900.0, 800.0],
            "weak_peak": [5.0, 4.0, 3.0],
        }
    )

    result = filter_peak_columns(
        dataframe=dataframe,
        signal_columns=["strong_peak", "weak_peak"],
        min_initial_intensity=100.0,
    )

    assert result.kept_columns == ["strong_peak"]
    assert result.removed_columns == ["weak_peak"]
    assert "initial intensity" in result.removal_reasons["weak_peak"]


def test_filter_peak_columns_removes_flat_peak():
    dataframe = pd.DataFrame(
        {
            "time": [0.0, 1.0, 2.0],
            "changing_peak": [1000.0, 900.0, 800.0],
            "flat_peak": [500.0, 500.0, 500.0],
        }
    )

    result = filter_peak_columns(
        dataframe=dataframe,
        signal_columns=["changing_peak", "flat_peak"],
        min_dynamic_range=10.0,
    )

    assert result.kept_columns == ["changing_peak"]
    assert result.removed_columns == ["flat_peak"]
    assert "dynamic range" in result.removal_reasons["flat_peak"]


def test_filter_peak_columns_rejects_if_all_columns_removed():
    dataframe = pd.DataFrame(
        {
            "time": [0.0, 1.0, 2.0],
            "flat_peak": [500.0, 500.0, 500.0],
        }
    )

    with pytest.raises(ValueError):
        filter_peak_columns(
            dataframe=dataframe,
            signal_columns=["flat_peak"],
            min_dynamic_range=10.0,
        )


def test_interpolate_missing_values():
    dataframe = pd.DataFrame(
        {
            "time": [0.0, 1.0, 2.0],
            "A23_HN": [1.0, np.nan, 0.0],
        }
    )

    cleaned = interpolate_missing_values(
        dataframe=dataframe,
        signal_columns=["A23_HN"],
    )

    assert list(cleaned["A23_HN"]) == [1.0, 0.5, 0.0]


def test_prepare_peak_dataframe_filters_and_interpolates():
    dataframe = pd.DataFrame(
        {
            "time": [0.0, 1.0, 2.0, 3.0],
            "good_peak": [1.0, np.nan, 0.5, 0.25],
            "too_missing": [1.0, np.nan, np.nan, np.nan],
            "flat_peak": [1.0, 1.0, 1.0, 1.0],
            "metadata": ["a", "b", "c", "d"],
        }
    )

    prepared, result = prepare_peak_dataframe(
        dataframe=dataframe,
        time_column="time",
        exclude_columns=["metadata"],
        max_missing_fraction=0.25,
        min_dynamic_range=0.1,
        interpolate_missing=True,
    )

    assert result.kept_columns == ["good_peak"]
    assert set(result.removed_columns) == {"too_missing", "flat_peak"}

    assert list(prepared.columns) == ["time", "good_peak"]
    assert not prepared["good_peak"].isna().any()
    assert prepared["good_peak"].iloc[1] == pytest.approx(0.75)


def test_prepare_peak_dataframe_raises_if_missing_remains_without_interpolation():
    dataframe = pd.DataFrame(
        {
            "time": [0.0, 1.0, 2.0],
            "A23_HN": [1.0, np.nan, 0.0],
        }
    )

    with pytest.raises(ValueError):
        prepare_peak_dataframe(
            dataframe=dataframe,
            time_column="time",
            signal_columns=["A23_HN"],
            max_missing_fraction=0.5,
            interpolate_missing=False,
        )


def test_build_peak_filtering_table():
    dataframe = pd.DataFrame(
        {
            "time": [0.0, 1.0, 2.0],
            "good_peak": [1.0, 0.8, 0.6],
            "flat_peak": [1.0, 1.0, 1.0],
        }
    )

    result = filter_peak_columns(
        dataframe=dataframe,
        signal_columns=["good_peak", "flat_peak"],
        min_dynamic_range=0.1,
    )

    table = build_peak_filtering_table(result)

    assert list(table.columns) == [
        "signal_column",
        "kept",
        "removal_reason",
    ]

    assert set(table["signal_column"]) == {"good_peak", "flat_peak"}
    assert table.loc[table["signal_column"] == "good_peak", "kept"].iloc[0]
    assert not table.loc[table["signal_column"] == "flat_peak", "kept"].iloc[0]
