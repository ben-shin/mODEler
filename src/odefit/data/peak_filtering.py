from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PeakFilteringResult:
    """
    Result from filtering wide-format peak/intensity columns.

    kept_columns:
        Signal columns that passed filtering.

    removed_columns:
        Signal columns that were removed.

    removal_reasons:
        Mapping from removed column name to human-readable reason.
    """

    kept_columns: list[str]
    removed_columns: list[str]
    removal_reasons: dict[str, str]


def infer_numeric_signal_columns(
    dataframe: pd.DataFrame,
    time_column: str,
    exclude_columns: list[str] | None = None,
) -> list[str]:
    """
    Infer numeric signal columns from a wide-format dataframe.

    The time column is excluded automatically.
    Extra metadata columns can be excluded with exclude_columns.

    Example:

        time,A23_HN,G45_HN,assignment
        0,1000,800,A23
        1,900,760,A23

    returns:

        ["A23_HN", "G45_HN"]
    """

    if time_column not in dataframe.columns:
        raise ValueError(f"Time column not found in dataframe: {time_column}")

    excluded = {time_column}

    if exclude_columns is not None:
        excluded.update(exclude_columns)

    signal_columns: list[str] = []

    for column in dataframe.columns:
        if column in excluded:
            continue

        if pd.api.types.is_numeric_dtype(dataframe[column]):
            signal_columns.append(column)

    if not signal_columns:
        raise ValueError("No numeric signal columns could be inferred")

    return signal_columns


def calculate_missing_fraction(
    dataframe: pd.DataFrame,
    column: str,
) -> float:
    """
    Calculate fraction of missing values in one column.
    """

    if column not in dataframe.columns:
        raise ValueError(f"Column not found in dataframe: {column}")

    return float(dataframe[column].isna().mean())


def calculate_initial_intensity(
    dataframe: pd.DataFrame,
    column: str,
    n_initial_points: int = 1,
    absolute: bool = True,
) -> float:
    """
    Calculate mean initial intensity using the first n_initial_points.

    If absolute is True, absolute intensities are used.
    """

    if column not in dataframe.columns:
        raise ValueError(f"Column not found in dataframe: {column}")

    if n_initial_points < 1:
        raise ValueError("n_initial_points must be at least 1")

    values = dataframe[column].iloc[:n_initial_points].astype(float)

    if absolute:
        values = values.abs()

    return float(values.mean(skipna=True))


def calculate_dynamic_range(
    dataframe: pd.DataFrame,
    column: str,
    absolute: bool = True,
) -> float:
    """
    Calculate dynamic range of one signal column.

    Dynamic range = max - min, ignoring NaNs.
    """

    if column not in dataframe.columns:
        raise ValueError(f"Column not found in dataframe: {column}")

    values = dataframe[column].astype(float)

    if absolute:
        values = values.abs()

    if values.dropna().empty:
        return 0.0

    return float(values.max(skipna=True) - values.min(skipna=True))


def filter_peak_columns(
    dataframe: pd.DataFrame,
    signal_columns: list[str],
    max_missing_fraction: float = 0.0,
    min_initial_intensity: float | None = None,
    initial_points: int = 1,
    min_dynamic_range: float | None = None,
) -> PeakFilteringResult:
    """
    Filter wide-format peak intensity columns.

    Removes columns when:
    - missing fraction is greater than max_missing_fraction
    - initial intensity is below min_initial_intensity
    - dynamic range is below min_dynamic_range

    This does not modify the dataframe.
    """

    if not 0.0 <= max_missing_fraction <= 1.0:
        raise ValueError("max_missing_fraction must be between 0 and 1")

    kept_columns: list[str] = []
    removed_columns: list[str] = []
    removal_reasons: dict[str, str] = {}

    for column in signal_columns:
        if column not in dataframe.columns:
            raise ValueError(f"Signal column not found in dataframe: {column}")

        missing_fraction = calculate_missing_fraction(
            dataframe=dataframe,
            column=column,
        )

        if missing_fraction > max_missing_fraction:
            removed_columns.append(column)
            removal_reasons[column] = (
                f"missing fraction {missing_fraction:.3f} exceeds "
                f"maximum {max_missing_fraction:.3f}"
            )
            continue

        if min_initial_intensity is not None:
            initial_intensity = calculate_initial_intensity(
                dataframe=dataframe,
                column=column,
                n_initial_points=initial_points,
                absolute=True,
            )

            if not np.isfinite(initial_intensity):
                removed_columns.append(column)
                removal_reasons[column] = "initial intensity is not finite"
                continue

            if initial_intensity < min_initial_intensity:
                removed_columns.append(column)
                removal_reasons[column] = (
                    f"initial intensity {initial_intensity:.6g} is below "
                    f"minimum {min_initial_intensity:.6g}"
                )
                continue

        if min_dynamic_range is not None:
            dynamic_range = calculate_dynamic_range(
                dataframe=dataframe,
                column=column,
                absolute=True,
            )

            if dynamic_range < min_dynamic_range:
                removed_columns.append(column)
                removal_reasons[column] = (
                    f"dynamic range {dynamic_range:.6g} is below "
                    f"minimum {min_dynamic_range:.6g}"
                )
                continue

        kept_columns.append(column)

    if not kept_columns:
        raise ValueError("All signal columns were removed by peak filtering")

    return PeakFilteringResult(
        kept_columns=kept_columns,
        removed_columns=removed_columns,
        removal_reasons=removal_reasons,
    )


def interpolate_missing_values(
    dataframe: pd.DataFrame,
    signal_columns: list[str],
) -> pd.DataFrame:
    """
    Linearly interpolate missing values in signal columns.

    Also fills leading/trailing NaNs using nearest available values.

    This is intended for kept peak columns with a small number of missing
    timepoints.
    """

    cleaned = dataframe.copy()

    for column in signal_columns:
        cleaned[column] = (
            cleaned[column]
            .astype(float)
            .interpolate(method="linear", limit_direction="both")
        )

    return cleaned


def ensure_no_missing_signal_values(
    dataframe: pd.DataFrame,
    signal_columns: list[str],
) -> None:
    """
    Raise an error if any selected signal columns still contain NaNs.
    """

    missing_columns = [
        column
        for column in signal_columns
        if dataframe[column].isna().any()
    ]

    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Signal columns still contain missing values: {missing}")


def prepare_peak_dataframe(
    dataframe: pd.DataFrame,
    time_column: str,
    signal_columns: list[str] | None = None,
    exclude_columns: list[str] | None = None,
    max_missing_fraction: float = 0.0,
    min_initial_intensity: float | None = None,
    initial_points: int = 1,
    min_dynamic_range: float | None = None,
    interpolate_missing: bool = True,
) -> tuple[pd.DataFrame, PeakFilteringResult]:
    """
    Prepare a wide-format peak dataframe for fitting.

    Steps:
    1. Infer signal columns if not provided.
    2. Filter bad peak columns.
    3. Keep only time + accepted signal columns.
    4. Optionally interpolate missing values in accepted signal columns.
    5. Verify no missing values remain.
    """

    if time_column not in dataframe.columns:
        raise ValueError(f"Time column not found in dataframe: {time_column}")

    if signal_columns is None:
        signal_columns = infer_numeric_signal_columns(
            dataframe=dataframe,
            time_column=time_column,
            exclude_columns=exclude_columns,
        )

    filtering_result = filter_peak_columns(
        dataframe=dataframe,
        signal_columns=signal_columns,
        max_missing_fraction=max_missing_fraction,
        min_initial_intensity=min_initial_intensity,
        initial_points=initial_points,
        min_dynamic_range=min_dynamic_range,
    )

    prepared = dataframe[
        [time_column] + filtering_result.kept_columns
    ].copy()

    if interpolate_missing:
        prepared = interpolate_missing_values(
            dataframe=prepared,
            signal_columns=filtering_result.kept_columns,
        )

    ensure_no_missing_signal_values(
        dataframe=prepared,
        signal_columns=filtering_result.kept_columns,
    )

    return prepared, filtering_result


def build_peak_filtering_table(
    filtering_result: PeakFilteringResult,
) -> pd.DataFrame:
    """
    Build a table describing peak filtering results.
    """

    rows: list[dict[str, str | bool]] = []

    for column in filtering_result.kept_columns:
        rows.append(
            {
                "signal_column": column,
                "kept": True,
                "removal_reason": "",
            }
        )

    for column in filtering_result.removed_columns:
        rows.append(
            {
                "signal_column": column,
                "kept": False,
                "removal_reason": filtering_result.removal_reasons[column],
            }
        )

    return pd.DataFrame(rows)
