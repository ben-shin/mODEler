from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_bootstrap_parameter_histograms(
    parameter_samples: pd.DataFrame,
    output_dir: str | Path,
    bins: int = 30,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    written_files: dict[str, Path] = {}

    parameter_columns = [
        column
        for column in parameter_samples.columns
        if column != "bootstrap_index"
    ]

    for parameter in parameter_columns:
        values = parameter_samples[parameter].dropna()

        fig, ax = plt.subplots()
        ax.hist(values, bins=bins)
        ax.set_xlabel(parameter)
        ax.set_ylabel("Count")
        ax.set_title(f"Bootstrap distribution: {parameter}")

        path = output_path / f"bootstrap_histogram_{parameter}.png"
        fig.savefig(path, bbox_inches="tight", dpi=300)
        plt.close(fig)

        written_files[f"histogram_{parameter}"] = path

    return written_files


def plot_bootstrap_parameter_pairs(
    parameter_samples: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    written_files: dict[str, Path] = {}

    parameter_columns = [
        column
        for column in parameter_samples.columns
        if column != "bootstrap_index"
    ]

    for i, x_parameter in enumerate(parameter_columns):
        for y_parameter in parameter_columns[i + 1 :]:
            fig, ax = plt.subplots()

            ax.scatter(
                parameter_samples[x_parameter],
                parameter_samples[y_parameter],
                s=20,
                alpha=0.7,
            )

            ax.set_xlabel(x_parameter)
            ax.set_ylabel(y_parameter)
            ax.set_title(
                f"Bootstrap parameter relationship: "
                f"{x_parameter} vs {y_parameter}"
            )

            path = (
                output_path
                / f"bootstrap_pair_{x_parameter}_vs_{y_parameter}.png"
            )

            fig.savefig(path, bbox_inches="tight", dpi=300)
            plt.close(fig)

            written_files[f"pair_{x_parameter}_vs_{y_parameter}"] = path

    return written_files


def plot_bootstrap_prediction_bands(
    *,
    original_dataframe: pd.DataFrame,
    bootstrap_prediction_dataframes: list[pd.DataFrame],
    time_column: str,
    signal_columns: list[str],
    output_dir: str | Path,
    confidence_level: float = 0.95,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    written_files: dict[str, Path] = {}

    alpha = 1.0 - confidence_level
    lower_q = alpha / 2.0
    upper_q = 1.0 - alpha / 2.0

    time = np.asarray(original_dataframe[time_column], dtype=float)

    for signal_column in signal_columns:
        predictions = np.asarray(
            [
                dataframe[signal_column].to_numpy(dtype=float)
                for dataframe in bootstrap_prediction_dataframes
                if signal_column in dataframe.columns
            ]
        )

        if predictions.size == 0:
            continue

        lower = np.quantile(predictions, lower_q, axis=0)
        median = np.quantile(predictions, 0.5, axis=0)
        upper = np.quantile(predictions, upper_q, axis=0)

        fig, ax = plt.subplots()

        ax.scatter(
            time,
            original_dataframe[signal_column],
            label="Observed",
            s=20,
        )

        ax.plot(
            time,
            median,
            label="Bootstrap median",
        )

        ax.fill_between(
            time,
            lower,
            upper,
            alpha=0.25,
            label=f"{confidence_level:.0%} interval",
        )

        ax.set_xlabel(time_column)
        ax.set_ylabel(signal_column)
        ax.set_title(f"Bootstrap prediction band: {signal_column}")
        ax.legend()

        path = output_path / f"bootstrap_prediction_band_{signal_column}.png"
        fig.savefig(path, bbox_inches="tight", dpi=300)
        plt.close(fig)

        written_files[f"prediction_band_{signal_column}"] = path

    return written_files
