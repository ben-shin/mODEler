from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


def parse_tau_from_column(column: str, fallback_index: int) -> float:
    """
    Try to recover lag time from a column name.

    Handles names like:
      tau_0.001
      G_0.001
      lag_1e-3
      0.001
    """

    text = str(column)

    try:
        return float(text)
    except ValueError:
        pass

    matches = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)

    if matches:
        return float(matches[-1])

    return float(fallback_index)


def infer_time_column(dataframe: pd.DataFrame, requested: str | None) -> str:
    if requested is not None:
        if requested not in dataframe.columns:
            raise ValueError(f"Requested time column not found: {requested}")
        return requested

    for candidate in ["time_min", "time", "Time", "tau_time", "elapsed_time", "t"]:
        if candidate in dataframe.columns:
            return candidate

    numeric = dataframe.select_dtypes(include="number").columns.tolist()

    if not numeric:
        raise ValueError("No numeric columns found.")

    return numeric[0]


def extract_features(
    dataframe: pd.DataFrame,
    *,
    time_column: str,
    early_points: int,
    late_points: int,
) -> pd.DataFrame:
    numeric_columns = dataframe.select_dtypes(include="number").columns.tolist()

    signal_columns = [
        column
        for column in numeric_columns
        if column != time_column
    ]

    if not signal_columns:
        raise ValueError("No numeric FCS signal columns found.")

    tau_values = np.array(
        [
            parse_tau_from_column(column, index)
            for index, column in enumerate(signal_columns)
        ],
        dtype=float,
    )

    order = np.argsort(tau_values)
    tau_values = tau_values[order]
    signal_columns = [signal_columns[index] for index in order]

    matrix = dataframe[signal_columns].to_numpy(dtype=float)

    n_tau = matrix.shape[1]

    early_n = min(max(early_points, 1), n_tau)
    late_n = min(max(late_points, 1), n_tau)

    early = np.nanmean(matrix[:, :early_n], axis=1)
    late = np.nanmean(matrix[:, -late_n:], axis=1)

    amplitude = early - late

    # Area under baseline-subtracted curve.
    baseline_subtracted = matrix - late[:, None]
    auc = np.trapezoid(baseline_subtracted, x=tau_values, axis=1)

    # Half-decay tau proxy.
    half_tau_values = []

    for row_index in range(matrix.shape[0]):
        curve = matrix[row_index, :]
        baseline = late[row_index]
        amp = amplitude[row_index]

        if not np.isfinite(amp) or amp == 0:
            half_tau_values.append(np.nan)
            continue

        target = baseline + 0.5 * amp

        # Assumes decay-ish curve. Finds closest point to half amplitude.
        index = int(np.nanargmin(np.abs(curve - target)))
        half_tau_values.append(tau_values[index])

    output = pd.DataFrame(
        {
            time_column: dataframe[time_column].to_numpy(dtype=float),
            "fcs_early_mean": early,
            "fcs_late_mean": late,
            "fcs_amplitude_proxy": amplitude,
            "fcs_auc_proxy": auc,
            "fcs_half_tau_proxy": np.asarray(half_tau_values, dtype=float),
        }
    )

    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract ODE-ready FCS feature timecourses from a raw FCS matrix."
        )
    )

    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--time-column", default=None)
    parser.add_argument("--early-points", type=int, default=5)
    parser.add_argument("--late-points", type=int, default=20)

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataframe = pd.read_csv(input_path)
    time_column = infer_time_column(dataframe, args.time_column)

    features = extract_features(
        dataframe,
        time_column=time_column,
        early_points=args.early_points,
        late_points=args.late_points,
    )

    features.to_csv(output_path, index=False)

    print("\nFCS feature extraction complete")
    print("===============================")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Time column: {time_column}")
    print(f"Rows: {len(features)}")
    print("\nColumns:")
    for column in features.columns:
        print(f"  - {column}")

    print("\nPreview:")
    print(features.head().to_string(index=False))


if __name__ == "__main__":
    main()
