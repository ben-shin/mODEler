from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_tau(column: str, fallback: int) -> float:
    try:
        return float(column)
    except ValueError:
        pass

    matches = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(column))

    if matches:
        return float(matches[-1])

    return float(fallback)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot raw FCS G(tau) curves at selected elapsed timepoints."
    )

    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--time-column", default="time_min")
    parser.add_argument("--n-curves", type=int, default=12)

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataframe = pd.read_csv(input_path)

    time_column = args.time_column

    if time_column not in dataframe.columns:
        raise ValueError(f"Missing time column: {time_column}")

    numeric_columns = dataframe.select_dtypes(include="number").columns.tolist()

    signal_columns = [
        column
        for column in numeric_columns
        if column != time_column
    ]

    tau_values = np.array(
        [
            parse_tau(column, index)
            for index, column in enumerate(signal_columns)
        ],
        dtype=float,
    )

    order = np.argsort(tau_values)
    tau_values = tau_values[order]
    signal_columns = [signal_columns[index] for index in order]

    indices = np.linspace(0, len(dataframe) - 1, args.n_curves)
    indices = sorted(set(int(round(index)) for index in indices))

    fig, ax = plt.subplots(figsize=(8, 5))

    for row_index in indices:
        time_value = dataframe.iloc[row_index][time_column]
        curve = dataframe.iloc[row_index][signal_columns].to_numpy(dtype=float)

        ax.plot(
            tau_values,
            curve,
            linewidth=1.0,
            label=f"{time_value:g} min",
        )

    ax.set_xscale("log")
    ax.set_title("Raw FCS curves G(tau) at selected elapsed times")
    ax.set_xlabel("tau / lag-time column")
    ax.set_ylabel("G(tau)")

    if len(indices) <= 12:
        ax.legend(fontsize="small")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)

    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
