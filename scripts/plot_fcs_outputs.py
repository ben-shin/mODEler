from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COMMON_TIME_COLUMNS = [
    "time_min",
    "time",
    "Time",
    "tau_time",
    "elapsed_time",
    "elapsed_min",
    "t",
]


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_")


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def parse_tau_from_column(column: str, fallback_index: int) -> float:
    """
    Parse lag time from FCS columns such as:

      G_tau_1p000000e-03_ms
      G_tau_1p024000e+01_ms

    The p notation is converted back to decimal notation.
    """

    text = str(column)

    if text.startswith("G_tau_"):
        text = text.removeprefix("G_tau_")

    if text.endswith("_ms"):
        text = text.removesuffix("_ms")

    text = text.replace("p", ".")

    try:
        return float(text)
    except ValueError:
        pass

    matches = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)

    if matches:
        return float(matches[-1])

    return float(fallback_index)


def infer_time_column(dataframe: pd.DataFrame, requested: str | None = None) -> str:
    if requested and requested in dataframe.columns:
        return requested

    # backend predicted outputs usually write the indep variable as "time"
    # even when the source data used a specific name like "time_min"
    fallback_candidates = []

    if requested in {"time_min", "elapsed_min", "elapsed_time"}:
        fallback_candidates.extend(["time", "t", "Time"])

    fallback_candidates.extend(COMMON_TIME_COLUMNS)

    for candidate in COMMON_TIME_COLUMNS:
        if candidate in dataframe.columns:
            return candidate

    numeric_columns = dataframe.select_dtypes(include="number").columns.tolist()

    if not numeric_columns:
        raise ValueError("No numeric columns found.")

    return numeric_columns[0]


def infer_raw_fcs_signal_columns(
    dataframe: pd.DataFrame,
    *,
    time_column: str,
) -> list[str]:
    numeric_columns = dataframe.select_dtypes(include="number").columns.tolist()

    g_tau_columns = [
        column
        for column in numeric_columns
        if str(column).startswith("G_tau_")
    ]

    if g_tau_columns:
        return g_tau_columns

    return [
        column
        for column in numeric_columns
        if column != time_column
    ]


def sort_fcs_columns_by_tau(signal_columns: list[str]) -> tuple[list[str], np.ndarray]:
    tau_values = np.array(
        [
            parse_tau_from_column(column, index)
            for index, column in enumerate(signal_columns)
        ],
        dtype=float,
    )

    order = np.argsort(tau_values)

    sorted_columns = [
        signal_columns[index]
        for index in order
    ]

    return sorted_columns, tau_values[order]


def get_numeric_signal_columns(
    dataframe: pd.DataFrame,
    *,
    time_column: str,
    requested: list[str] | None = None,
) -> list[str]:
    if requested:
        missing = [
            column
            for column in requested
            if column not in dataframe.columns
        ]

        if missing:
            raise ValueError(f"Requested signal columns missing: {missing}")

        return requested

    numeric_columns = dataframe.select_dtypes(include="number").columns.tolist()

    return [
        column
        for column in numeric_columns
        if column != time_column
    ]


def choose_evenly_spaced_indices(n_items: int, n_select: int) -> list[int]:
    if n_items <= 0:
        return []

    if n_items <= n_select:
        return list(range(n_items))

    indices = np.linspace(0, n_items - 1, n_select)
    return sorted(set(int(round(index)) for index in indices))


def compute_ic_weights(
    comparison_table: pd.DataFrame,
    *,
    criterion: str,
) -> pd.DataFrame:
    if criterion not in comparison_table.columns:
        raise ValueError(f"Comparison table does not contain {criterion!r}.")

    table = comparison_table.copy()

    if "success" in table.columns:
        table = table[table["success"].astype(bool)].copy()

    table[criterion] = pd.to_numeric(table[criterion], errors="coerce")
    table = table.dropna(subset=[criterion]).copy()

    if table.empty:
        raise ValueError(f"No successful rows with numeric {criterion}.")

    delta = table[criterion].astype(float) - float(table[criterion].min())
    raw_weights = np.exp(-0.5 * delta.to_numpy(dtype=float))
    weights = raw_weights / raw_weights.sum()

    table[f"delta_{criterion}"] = delta
    table[f"{criterion}_weight"] = weights
    table[f"{criterion}_weight_percent"] = 100.0 * weights

    table = table.sort_values(
        f"{criterion}_weight_percent",
        ascending=False,
    ).reset_index(drop=True)

    table.insert(0, f"{criterion}_weight_rank", range(1, len(table) + 1))

    return table

def plot_feature_fit_profiles(
    *,
    output_dir: Path,
    figures_dir: Path,
    top_n: int,
    sort_by: str,
    log_x: bool,
) -> list[Path]:
    """
    Plot observed FCS-derived feature timecourses against fitted model functions.

    This is the correct plot for feature-extracted FCS fitting, e.g.:

        fcs_amplitude_proxy(time_min) observed
        vs
        fcs_amplitude_proxy(time) fitted
    """

    config = read_generated_config(output_dir)
    comparison_table = read_comparison_table(output_dir)

    data_path = Path(config["data"])

    if not data_path.exists():
        raise FileNotFoundError(f"Original fitted data path not found: {data_path}")

    observed = pd.read_csv(data_path)

    observed_time_column = infer_time_column(
        observed,
        config.get("time_column"),
    )

    signal_columns = get_numeric_signal_columns(
        observed,
        time_column=observed_time_column,
        requested=config.get("signal_columns"),
    )

    model_names = choose_top_models(
        comparison_table,
        top_n=top_n,
        sort_by=sort_by,
    )

    figures_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []

    observed_x = observed[observed_time_column].to_numpy(dtype=float)

    # One combined figure per feature, comparing top models.
    feature_dir = figures_dir / "feature_fit_profiles"
    feature_dir.mkdir(parents=True, exist_ok=True)

    for signal_column in signal_columns:
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(
            observed_x,
            observed[signal_column].to_numpy(dtype=float),
            marker="o",
            markersize=4,
            linewidth=1.2,
            label="observed",
        )

        for model_name in model_names:
            _, predicted, _, _, _ = read_model_outputs(output_dir, model_name)

            predicted_time_column = infer_time_column(
                predicted,
                observed_time_column,
            )

            if signal_column not in predicted.columns:
                continue

            predicted_x = predicted[predicted_time_column].to_numpy(dtype=float)

            ax.plot(
                predicted_x,
                predicted[signal_column].to_numpy(dtype=float),
                linewidth=1.8,
                label=f"{model_name} fitted",
            )

        if log_x and np.any(observed_x > 0):
            ax.set_xscale("log")

        ax.set_title(f"{signal_column}: observed FCS feature vs fitted functions")
        ax.set_xlabel(observed_time_column + (" [log]" if log_x else ""))
        ax.set_ylabel(signal_column)
        ax.legend(fontsize="small")

        fig.tight_layout()

        path = feature_dir / f"{safe_name(signal_column)}_observed_vs_fitted_models.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)

        written.append(path)

    # One combined panel-like overlay for all features for the best model.
    best_model = model_names[0]
    _, best_predicted, _, _, _ = read_model_outputs(output_dir, best_model)

    best_predicted_time_column = infer_time_column(
        best_predicted,
        observed_time_column,
    )

    best_predicted_x = best_predicted[best_predicted_time_column].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(9, 6))

    for signal_column in signal_columns:
        if signal_column not in best_predicted.columns:
            continue

        ax.plot(
            observed_x,
            observed[signal_column].to_numpy(dtype=float),
            marker="o",
            markersize=3,
            linewidth=0.9,
            alpha=0.65,
            label=f"{signal_column} observed",
        )

        ax.plot(
            best_predicted_x,
            best_predicted[signal_column].to_numpy(dtype=float),
            linewidth=1.6,
            alpha=0.90,
            label=f"{signal_column} fitted",
        )

    if log_x and np.any(observed_x > 0):
        ax.set_xscale("log")

    ax.set_title(f"Best model {best_model}: FCS feature profiles vs fitted functions")
    ax.set_xlabel(observed_time_column + (" [log]" if log_x else ""))
    ax.set_ylabel("feature value")

    if len(signal_columns) <= 4:
        ax.legend(fontsize="small")

    fig.tight_layout()

    path = feature_dir / f"best_model_{safe_name(best_model)}_all_features_observed_vs_fitted.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)

    written.append(path)

    # Write a small index.
    index_path = feature_dir / "feature_fit_profile_index.txt"

    with index_path.open("w") as handle:
        handle.write("FCS feature fit profile plots\n")
        handle.write("=============================\n\n")
        handle.write(f"Output dir: {output_dir}\n")
        handle.write(f"Top models: {', '.join(model_names)}\n")
        handle.write(f"Observed time column: {observed_time_column}\n")
        handle.write(f"Signal columns: {', '.join(signal_columns)}\n\n")

        for path in written:
            handle.write(f"- {path.name}\n")

    written.append(index_path)

    return written

def read_generated_config(output_dir: Path) -> dict[str, Any]:
    path = output_dir / "generated_fcs_all_txt_config.json"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing generated config: {path}. "
            "Rerun scripts/run_fcs_txt_models.py first."
        )

    return load_json(path)


def read_comparison_table(output_dir: Path) -> pd.DataFrame:
    path = output_dir / "fcs_all_txt_model_comparison_table.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing comparison table: {path}. "
            "Rerun scripts/run_fcs_txt_models.py first."
        )

    return pd.read_csv(path)


def read_model_outputs(output_dir: Path, model_name: str):
    model_dir = output_dir / "models" / safe_name(model_name)

    predicted_path = model_dir / "predicted.csv"
    residuals_path = model_dir / "residuals.csv"
    observable_path = model_dir / "observable_table.csv"
    summary_path = model_dir / "fit_summary.json"

    if not predicted_path.exists():
        raise FileNotFoundError(
            f"Missing predicted file: {predicted_path}. "
            "Patch/rerun scripts/run_fcs_txt_models.py so it exports per-model outputs."
        )

    predicted = pd.read_csv(predicted_path)

    residuals = (
        pd.read_csv(residuals_path)
        if residuals_path.exists()
        else pd.DataFrame()
    )

    observable = (
        pd.read_csv(observable_path)
        if observable_path.exists()
        else pd.DataFrame()
    )

    summary = (
        load_json(summary_path)
        if summary_path.exists()
        else {}
    )

    return model_dir, predicted, residuals, observable, summary


def plot_raw_fcs_curves(
    *,
    input_path: Path,
    output_dir: Path,
    time_column: str | None,
    n_curves: int,
    log_x: bool,
) -> list[Path]:
    dataframe = pd.read_csv(input_path)

    time_column = infer_time_column(dataframe, time_column)

    signal_columns = infer_raw_fcs_signal_columns(
        dataframe,
        time_column=time_column,
    )

    signal_columns, tau_values = sort_fcs_columns_by_tau(signal_columns)

    row_indices = choose_evenly_spaced_indices(len(dataframe), n_curves)

    output_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []

    fig, ax = plt.subplots(figsize=(8, 5))

    for row_index in row_indices:
        row = dataframe.iloc[row_index]
        time_value = row[time_column]
        curve = row[signal_columns].to_numpy(dtype=float)

        ax.plot(
            tau_values,
            curve,
            linewidth=1.0,
            label=f"{time_value:g} min",
        )

    if log_x:
        positive = tau_values > 0

        if positive.any():
            ax.set_xscale("log")

    ax.set_title("Raw FCS curves G(tau) at selected elapsed times")
    ax.set_xlabel("tau / lag time (ms)" + (" [log]" if log_x else ""))
    ax.set_ylabel("G(tau)")

    if len(row_indices) <= 12:
        ax.legend(fontsize="small")

    fig.tight_layout()

    output_path = output_dir / "raw_fcs_curves_by_elapsed_time.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    written.append(output_path)

    # Also write a tau parser preview for sanity checking.
    tau_preview = pd.DataFrame(
        {
            "column": signal_columns,
            "tau_ms": tau_values,
        }
    )
    tau_preview_path = output_dir / "raw_fcs_tau_column_map.csv"
    tau_preview.to_csv(tau_preview_path, index=False)
    written.append(tau_preview_path)

    return written


def plot_feature_timecourses(
    *,
    input_path: Path,
    output_dir: Path,
    time_column: str | None,
    signal_columns: list[str] | None,
    log_x: bool,
) -> list[Path]:
    dataframe = pd.read_csv(input_path)

    time_column = infer_time_column(dataframe, time_column)

    signal_columns = get_numeric_signal_columns(
        dataframe,
        time_column=time_column,
        requested=signal_columns,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []

    x = dataframe[time_column].to_numpy(dtype=float)

    # Combined feature plot.
    fig, ax = plt.subplots(figsize=(9, 5))

    for column in signal_columns:
        ax.plot(
            x,
            dataframe[column].to_numpy(dtype=float),
            marker="o",
            markersize=3,
            linewidth=1.0,
            label=column,
        )

    if log_x and np.any(x > 0):
        ax.set_xscale("log")

    ax.set_title("FCS-derived feature timecourses")
    ax.set_xlabel(time_column + (" [log]" if log_x else ""))
    ax.set_ylabel("feature value")

    if len(signal_columns) <= 12:
        ax.legend(fontsize="small")

    fig.tight_layout()

    output_path = output_dir / "fcs_feature_timecourses_combined.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    written.append(output_path)

    # Individual feature plots.
    individual_dir = output_dir / "feature_timecourses"
    individual_dir.mkdir(parents=True, exist_ok=True)

    for column in signal_columns:
        fig, ax = plt.subplots(figsize=(7, 4.5))

        ax.plot(
            x,
            dataframe[column].to_numpy(dtype=float),
            marker="o",
            markersize=3,
            linewidth=1.3,
        )

        if log_x and np.any(x > 0):
            ax.set_xscale("log")

        ax.set_title(column)
        ax.set_xlabel(time_column + (" [log]" if log_x else ""))
        ax.set_ylabel("feature value")

        fig.tight_layout()

        path = individual_dir / f"{safe_name(column)}.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        written.append(path)

    return written


def plot_rankings(
    *,
    comparison_table: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []

    for criterion in ["bic", "aic", "rss"]:
        if criterion not in comparison_table.columns:
            continue

        table = comparison_table.copy()
        table[criterion] = pd.to_numeric(table[criterion], errors="coerce")
        table = table.dropna(subset=[criterion]).copy()

        if table.empty:
            continue

        table = table.sort_values(criterion)

        fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(table)), 5))

        ax.bar(
            table["model_name"].astype(str),
            table[criterion].astype(float),
        )

        ax.set_title(f"Model ranking by {criterion.upper()}")
        ax.set_xlabel("model")
        ax.set_ylabel(criterion.upper())
        ax.tick_params(axis="x", rotation=45)

        fig.tight_layout()

        path = output_dir / f"model_ranking_{criterion}.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        written.append(path)

    return written


def plot_model_weights(
    *,
    comparison_table: pd.DataFrame,
    output_dir: Path,
    criteria: list[str],
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    summary = comparison_table.copy()

    for criterion in criteria:
        if criterion not in comparison_table.columns:
            continue

        weights = compute_ic_weights(
            comparison_table,
            criterion=criterion,
        )

        weights_path = output_dir / f"model_{criterion}_weights.csv"
        weights.to_csv(weights_path, index=False)
        written.append(weights_path)

        keep = [
            "model_name",
            f"delta_{criterion}",
            f"{criterion}_weight",
            f"{criterion}_weight_percent",
        ]

        summary = summary.merge(
            weights[keep],
            on="model_name",
            how="left",
        )

        fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(weights)), 5))

        ax.bar(
            weights["model_name"].astype(str),
            weights[f"{criterion}_weight_percent"].astype(float),
        )

        ax.set_title(f"Relative model support from {criterion.upper()} weights")
        ax.set_xlabel("model")
        ax.set_ylabel("relative model support (%)")
        ax.tick_params(axis="x", rotation=45)

        fig.tight_layout()

        path = output_dir / f"model_{criterion}_weight_percent.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        written.append(path)

    summary_path = output_dir / "model_weight_summary.csv"
    summary.to_csv(summary_path, index=False)
    written.append(summary_path)

    return written


def choose_top_models(
    comparison_table: pd.DataFrame,
    *,
    top_n: int,
    sort_by: str,
) -> list[str]:
    table = comparison_table.copy()

    if "success" in table.columns:
        table = table[table["success"].astype(bool)].copy()

    if sort_by in table.columns:
        table[sort_by] = pd.to_numeric(table[sort_by], errors="coerce")
        table = table.sort_values(sort_by, na_position="last")

    if table.empty:
        raise ValueError("No successful models available for plotting.")

    return [
        str(value)
        for value in table["model_name"].head(top_n)
    ]


def plot_fit_overlays_for_model(
    *,
    output_dir: Path,
    observed: pd.DataFrame,
    model_name: str,
    time_column: str,
    signal_columns: list[str],
    figures_dir: Path,
    max_individual_plots: int | None,
    max_combined_traces: int,
    log_x: bool,
) -> list[Path]:
    _, predicted, residuals, observable, summary = read_model_outputs(
        output_dir,
        model_name,
    )

    written: list[Path] = []

    model_dir = figures_dir / safe_name(model_name)
    model_dir.mkdir(parents=True, exist_ok=True)

    observed_time = observed[time_column].to_numpy(dtype=float)
    predicted_time_column = infer_time_column(predicted, time_column)
    predicted_time = predicted[predicted_time_column].to_numpy(dtype=float)

    # Individual observed-vs-fitted traces.
    individual_dir = model_dir / "individual_timecourses"
    individual_dir.mkdir(parents=True, exist_ok=True)

    columns_to_plot = signal_columns

    if max_individual_plots is not None:
        columns_to_plot = columns_to_plot[:max_individual_plots]

    for column in columns_to_plot:
        if column not in observed.columns or column not in predicted.columns:
            continue

        fig, ax = plt.subplots(figsize=(7, 4.5))

        ax.plot(
            observed_time,
            observed[column].to_numpy(dtype=float),
            marker="o",
            markersize=3,
            linewidth=1.0,
            label="observed",
        )

        ax.plot(
            predicted_time,
            predicted[column].to_numpy(dtype=float),
            linewidth=1.8,
            label="fitted",
        )

        if log_x and np.any(observed_time > 0):
            ax.set_xscale("log")

        ax.set_title(f"{model_name}: {column}")
        ax.set_xlabel(time_column + (" [log]" if log_x else ""))
        ax.set_ylabel("signal")
        ax.legend()

        fig.tight_layout()

        path = individual_dir / f"{safe_name(column)}.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        written.append(path)

    # Combined observed/fitted traces for model.
    combined_columns = signal_columns

    if len(combined_columns) > max_combined_traces:
        indices = choose_evenly_spaced_indices(
            len(combined_columns),
            max_combined_traces,
        )
        combined_columns = [
            combined_columns[index]
            for index in indices
        ]

    fig, ax = plt.subplots(figsize=(10, 6))

    for column in combined_columns:
        if column in observed.columns:
            ax.plot(
                observed_time,
                observed[column].to_numpy(dtype=float),
                linewidth=0.7,
                alpha=0.30,
            )

    for column in combined_columns:
        if column in predicted.columns:
            ax.plot(
                predicted_time,
                predicted[column].to_numpy(dtype=float),
                linewidth=1.0,
                alpha=0.85,
            )

    if log_x and np.any(observed_time > 0):
        ax.set_xscale("log")

    ax.set_title(f"{model_name}: combined observed and fitted timecourses")
    ax.set_xlabel(time_column + (" [log]" if log_x else ""))
    ax.set_ylabel("signal")

    fig.tight_layout()

    path = model_dir / "combined_observed_vs_fitted.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    written.append(path)

    # Residual heatmap.
    if not residuals.empty:
        residual_time_column = infer_time_column(residuals, time_column)
        available = [
            column
            for column in signal_columns
            if column in residuals.columns
        ]

        if available:
            matrix = residuals[available].to_numpy(dtype=float).T

            fig, ax = plt.subplots(figsize=(10, max(5, 0.03 * len(available))))

            image = ax.imshow(
                matrix,
                aspect="auto",
                interpolation="nearest",
            )

            ax.set_title(f"{model_name}: residual heatmap")
            ax.set_xlabel("time index")
            ax.set_ylabel("signal column index")

            fig.colorbar(image, ax=ax, label="residual")
            fig.tight_layout()

            path = model_dir / "residual_heatmap.png"
            fig.savefig(path, dpi=200)
            plt.close(fig)
            written.append(path)

    # Observable RSS.
    if not observable.empty and "rss" in observable.columns:
        label_column = (
            "data_column"
            if "data_column" in observable.columns
            else "signal_column"
            if "signal_column" in observable.columns
            else None
        )

        plot_df = observable.copy()

        if label_column is None:
            plot_df["signal_column"] = np.arange(len(plot_df)).astype(str)
            label_column = "signal_column"

        plot_df["rss"] = pd.to_numeric(plot_df["rss"], errors="coerce")
        plot_df = plot_df.dropna(subset=["rss"])
        plot_df = plot_df.sort_values("rss", ascending=False).head(50)

        if not plot_df.empty:
            fig, ax = plt.subplots(figsize=(12, 5))

            ax.bar(
                plot_df[label_column].astype(str),
                plot_df["rss"].astype(float),
            )

            ax.set_title(f"{model_name}: worst observable RSS values")
            ax.set_xlabel("signal column")
            ax.set_ylabel("RSS")
            ax.tick_params(axis="x", rotation=90, labelsize=6)

            fig.tight_layout()

            path = model_dir / "observable_rss_top50.png"
            fig.savefig(path, dpi=200)
            plt.close(fig)
            written.append(path)

    # Parameters.
    parameters = summary.get("fitted_parameters", {})

    if isinstance(parameters, dict) and parameters:
        names = list(parameters)
        values = [float(parameters[name]) for name in names]

        fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(names)), 4))

        ax.bar(names, values)
        ax.set_title(f"{model_name}: fitted parameters")
        ax.set_xlabel("parameter")
        ax.set_ylabel("value")
        ax.tick_params(axis="x", rotation=45)

        fig.tight_layout()

        path = model_dir / "parameter_values.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        written.append(path)

    return written


def plot_fit_outputs(
    *,
    output_dir: Path,
    figures_dir: Path,
    top_n: int,
    sort_by: str,
    criteria: list[str],
    max_individual_plots: int | None,
    max_combined_traces: int,
    log_x: bool,
) -> list[Path]:
    config = read_generated_config(output_dir)
    comparison_table = read_comparison_table(output_dir)

    data_path = Path(config["data"])

    if not data_path.exists():
        raise FileNotFoundError(f"Original data path not found: {data_path}")

    observed = pd.read_csv(data_path)

    time_column = infer_time_column(
        observed,
        config.get("time_column"),
    )

    signal_columns = get_numeric_signal_columns(
        observed,
        time_column=time_column,
        requested=config.get("signal_columns"),
    )

    figures_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []

    written.extend(
        plot_rankings(
            comparison_table=comparison_table,
            output_dir=figures_dir,
        )
    )

    written.extend(
        plot_model_weights(
            comparison_table=comparison_table,
            output_dir=figures_dir,
            criteria=criteria,
        )
    )

    top_models = choose_top_models(
        comparison_table,
        top_n=top_n,
        sort_by=sort_by,
    )

    for model_name in top_models:
        written.extend(
            plot_fit_overlays_for_model(
                output_dir=output_dir,
                observed=observed,
                model_name=model_name,
                time_column=time_column,
                signal_columns=signal_columns,
                figures_dir=figures_dir,
                max_individual_plots=max_individual_plots,
                max_combined_traces=max_combined_traces,
                log_x=log_x,
            )
        )

    return written


def write_index(
    *,
    output_path: Path,
    title: str,
    written_files: list[Path],
    root: Path,
) -> None:
    with output_path.open("w") as handle:
        handle.write(title + "\n")
        handle.write("=" * len(title) + "\n\n")

        for path in sorted(written_files):
            try:
                relative = path.relative_to(root)
            except ValueError:
                relative = path

            handle.write(f"- {relative}\n")


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--log-x", action="store_true", default=False)
    parser.add_argument("--no-log-x", action="store_true", default=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified FCS plotting utility."
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    raw_parser = subparsers.add_parser(
        "raw",
        help="Plot raw FCS G(tau) curves at selected elapsed times.",
    )
    raw_parser.add_argument("--input", required=True)
    raw_parser.add_argument("--output-dir", required=True)
    raw_parser.add_argument("--time-column", default=None)
    raw_parser.add_argument("--n-curves", type=int, default=12)
    raw_parser.add_argument("--log-x", action="store_true", default=True)
    raw_parser.add_argument("--no-log-x", action="store_true")

    features_parser = subparsers.add_parser(
        "features",
        help="Plot extracted FCS feature timecourses.",
    )
    features_parser.add_argument("--input", required=True)
    features_parser.add_argument("--output-dir", required=True)
    features_parser.add_argument("--time-column", default=None)
    features_parser.add_argument("--signal-columns", nargs="+", default=None)
    features_parser.add_argument("--log-x", action="store_true", default=False)
    features_parser.add_argument("--no-log-x", action="store_true")

    fits_parser = subparsers.add_parser(
        "fits",
        help="Plot model fitting outputs from run_fcs_txt_models.py.",
    )
    fits_parser.add_argument("--output-dir", required=True)
    fits_parser.add_argument("--figures-dir", default=None)
    fits_parser.add_argument("--top-n", type=int, default=3)
    fits_parser.add_argument("--sort-by", default="bic")
    fits_parser.add_argument("--criteria", nargs="+", default=["bic", "aic"])
    fits_parser.add_argument("--max-individual-plots", type=int, default=None)
    fits_parser.add_argument("--max-combined-traces", type=int, default=80)
    fits_parser.add_argument("--log-x", action="store_true", default=False)
    fits_parser.add_argument("--no-log-x", action="store_true")

    profile_parser = subparsers.add_parser(
        "profiles",
        help="Plot FCS-derived feature timecourses against fitted model functions.",
    )
    profile_parser.add_argument("--output-dir", required=True)
    profile_parser.add_argument("--figures-dir", default=None)
    profile_parser.add_argument("--top-n", type=int, default=3)
    profile_parser.add_argument("--sort-by", default="bic")
    profile_parser.add_argument("--log-x", action="store_true", default=False)
    profile_parser.add_argument("--no-log-x", action="store_true")

    all_parser = subparsers.add_parser(
        "all",
        help="Run raw/features/fits plotting where matching inputs are provided.",
    )

    all_parser.add_argument("--raw-input", default=None)
    all_parser.add_argument("--features-input", default=None)
    all_parser.add_argument("--fit-output-dir", default=None)
    all_parser.add_argument("--output-dir", required=True)
    all_parser.add_argument("--time-column", default=None)
    all_parser.add_argument("--feature-signal-columns", nargs="+", default=None)
    all_parser.add_argument("--n-raw-curves", type=int, default=12)
    all_parser.add_argument("--top-n", type=int, default=3)
    all_parser.add_argument("--sort-by", default="bic")
    all_parser.add_argument("--criteria", nargs="+", default=["bic", "aic"])
    all_parser.add_argument("--max-individual-plots", type=int, default=None)
    all_parser.add_argument("--max-combined-traces", type=int, default=80)
    all_parser.add_argument("--raw-log-x", action="store_true", default=True)
    all_parser.add_argument("--feature-log-x", action="store_true", default=False)
    all_parser.add_argument("--fit-log-x", action="store_true", default=False)

    args = parser.parse_args()

    written: list[Path] = []

    if args.command == "raw":
        output_dir = Path(args.output_dir)
        log_x = bool(args.log_x and not args.no_log_x)

        written = plot_raw_fcs_curves(
            input_path=Path(args.input),
            output_dir=output_dir,
            time_column=args.time_column,
            n_curves=args.n_curves,
            log_x=log_x,
        )

        index_path = output_dir / "figure_index.txt"
        write_index(
            output_path=index_path,
            title="Raw FCS figures",
            written_files=written,
            root=output_dir,
        )
        written.append(index_path)

    elif args.command == "features":
        output_dir = Path(args.output_dir)
        log_x = bool(args.log_x and not args.no_log_x)

        written = plot_feature_timecourses(
            input_path=Path(args.input),
            output_dir=output_dir,
            time_column=args.time_column,
            signal_columns=args.signal_columns,
            log_x=log_x,
        )

        index_path = output_dir / "figure_index.txt"
        write_index(
            output_path=index_path,
            title="FCS feature figures",
            written_files=written,
            root=output_dir,
        )
        written.append(index_path)

    elif args.command == "fits":
        output_dir = Path(args.output_dir)
        figures_dir = (
            Path(args.figures_dir)
            if args.figures_dir
            else output_dir / "figures"
        )
        log_x = bool(args.log_x and not args.no_log_x)

        written = plot_fit_outputs(
            output_dir=output_dir,
            figures_dir=figures_dir,
            top_n=args.top_n,
            sort_by=args.sort_by,
            criteria=args.criteria,
            max_individual_plots=args.max_individual_plots,
            max_combined_traces=args.max_combined_traces,
            log_x=log_x,
        )

        index_path = figures_dir / "figure_index.txt"
        write_index(
            output_path=index_path,
            title="FCS fit figures",
            written_files=written,
            root=figures_dir,
        )
        written.append(index_path)

    elif args.command == "profiles":
        output_dir = Path(args.output_dir)
        figures_dir = (
            Path(args.figures_dir)
            if args.figures_dir
            else output_dir / "figures"
        )
        log_x = bool(args.log_x and not args.no_log_x)

        written = plot_feature_fit_profiles(
            output_dir=output_dir,
            figures_dir=figures_dir,
            top_n=args.top_n,
            sort_by=args.sort_by,
            log_x=log_x,
        )

        index_path = figures_dir / "profile_figure_index.txt"
        write_index(
            output_path=index_path,
            title="FCS feature fit profile figures",
            written_files=written,
            root=figures_dir,
        )
        written.append(index_path)

    elif args.command == "all":
        root = Path(args.output_dir)
        root.mkdir(parents=True, exist_ok=True)

        if args.raw_input:
            written.extend(
                plot_raw_fcs_curves(
                    input_path=Path(args.raw_input),
                    output_dir=root / "raw",
                    time_column=args.time_column,
                    n_curves=args.n_raw_curves,
                    log_x=args.raw_log_x,
                )
            )

        if args.features_input:
            written.extend(
                plot_feature_timecourses(
                    input_path=Path(args.features_input),
                    output_dir=root / "features",
                    time_column=args.time_column,
                    signal_columns=args.feature_signal_columns,
                    log_x=args.feature_log_x,
                )
            )

        if args.fit_output_dir:
            written.extend(
                plot_fit_outputs(
                    output_dir=Path(args.fit_output_dir),
                    figures_dir=root / "fits",
                    top_n=args.top_n,
                    sort_by=args.sort_by,
                    criteria=args.criteria,
                    max_individual_plots=args.max_individual_plots,
                    max_combined_traces=args.max_combined_traces,
                    log_x=args.fit_log_x,
                )
            )

        index_path = root / "figure_index.txt"
        write_index(
            output_path=index_path,
            title="Unified FCS figure index",
            written_files=written,
            root=root,
        )
        written.append(index_path)

    print("\nFCS plotting complete")
    print("=====================")
    print(f"Figures/files written: {len(written)}")

    for path in written[:30]:
        print(f"  {path}")

    if len(written) > 30:
        print(f"  ... and {len(written) - 30} more")


if __name__ == "__main__":
    main()
