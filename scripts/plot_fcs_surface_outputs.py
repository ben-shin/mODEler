from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_")


def load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def parse_tau_from_column(column: str, fallback_index: int) -> float:
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


def read_run_config(output_dir: Path) -> dict:
    path = output_dir / "surface_fit_run_config.json"

    if not path.exists():
        raise FileNotFoundError(f"Missing run config: {path}")

    return load_json(path)


def read_comparison_table(output_dir: Path) -> pd.DataFrame:
    path = output_dir / "fcs_surface_model_comparison_table.csv"

    if not path.exists():
        raise FileNotFoundError(f"Missing comparison table: {path}")

    return pd.read_csv(path)


def choose_top_models(table: pd.DataFrame, *, top_n: int, sort_by: str) -> list[str]:
    working = table.copy()

    if "success" in working.columns:
        working = working[working["success"].astype(bool)]

    if sort_by in working.columns:
        working = working.sort_values(sort_by, na_position="last")

    if working.empty:
        raise ValueError("No successful models available to plot.")

    return [
        str(value)
        for value in working["model_name"].head(top_n)
    ]


def load_model_surface(output_dir: Path, model_name: str):
    model_dir = output_dir / "models" / safe_name(model_name)

    predicted_path = model_dir / "surface_predicted.csv"
    residuals_path = model_dir / "surface_residuals.csv"
    summary_path = model_dir / "surface_fit_summary.json"
    tau_map_path = model_dir / "tau_map.csv"

    if not predicted_path.exists():
        raise FileNotFoundError(f"Missing predicted surface: {predicted_path}")

    predicted = pd.read_csv(predicted_path)
    residuals = pd.read_csv(residuals_path)
    summary = load_json(summary_path) if summary_path.exists() else {}
    tau_map = pd.read_csv(tau_map_path) if tau_map_path.exists() else None

    return model_dir, predicted, residuals, summary, tau_map


def get_surface_matrix(dataframe: pd.DataFrame, *, time_column: str = "time_min"):
    signal_columns = [
        column
        for column in dataframe.columns
        if column != time_column
    ]

    tau_values = np.array(
        [
            parse_tau_from_column(column, index)
            for index, column in enumerate(signal_columns)
        ],
        dtype=float,
    )

    order = np.argsort(tau_values)

    tau_values = tau_values[order]
    signal_columns = [
        signal_columns[index]
        for index in order
    ]

    time_values = dataframe[time_column].to_numpy(dtype=float)
    matrix = dataframe[signal_columns].to_numpy(dtype=float)

    return time_values, tau_values, matrix


def plot_model_weights(table: pd.DataFrame, figures_dir: Path) -> list[Path]:
    written = []

    for criterion in ["bic", "aic"]:
        column = f"{criterion}_weight_percent"

        if column not in table.columns:
            continue

        working = table.dropna(subset=[column]).copy()

        if working.empty:
            continue

        working = working.sort_values(column, ascending=False)

        fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(working)), 5))

        ax.bar(
            working["model_name"].astype(str),
            working[column].astype(float),
        )

        ax.set_title(f"Relative model support from {criterion.upper()} weights")
        ax.set_xlabel("model")
        ax.set_ylabel("relative support (%)")
        ax.tick_params(axis="x", rotation=45)

        fig.tight_layout()

        path = figures_dir / f"surface_model_{criterion}_weight_percent.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)

        written.append(path)

    return written


def plot_surface_heatmap(
    *,
    matrix: np.ndarray,
    time_values: np.ndarray,
    tau_values: np.ndarray,
    output_path: Path,
    title: str,
    label: str,
):
    fig, ax = plt.subplots(figsize=(9, 6))

    positive_tau = tau_values[tau_values > 0]

    if len(positive_tau) == len(tau_values):
        extent = [
            np.log10(tau_values.min()),
            np.log10(tau_values.max()),
            time_values.min(),
            time_values.max(),
        ]
        x_label = "log10(tau / ms)"
    else:
        extent = [
            0,
            matrix.shape[1] - 1,
            time_values.min(),
            time_values.max(),
        ]
        x_label = "tau index"

    image = ax.imshow(
        matrix,
        aspect="auto",
        origin="lower",
        extent=extent,
        interpolation="nearest",
    )

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("time_min")

    fig.colorbar(image, ax=ax, label=label)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_profiles_at_selected_times(
    *,
    observed_matrix: np.ndarray,
    predicted_matrix: np.ndarray,
    time_values: np.ndarray,
    tau_values: np.ndarray,
    model_name: str,
    output_path: Path,
    n_profiles: int,
):
    indices = np.linspace(0, len(time_values) - 1, min(n_profiles, len(time_values)))
    indices = sorted(set(int(round(index)) for index in indices))

    fig, ax = plt.subplots(figsize=(8, 5))

    for index in indices:
        ax.plot(
            tau_values,
            observed_matrix[index, :],
            marker="o",
            markersize=2,
            linewidth=0.8,
            alpha=0.55,
            label=f"{time_values[index]:g} min observed",
        )

        ax.plot(
            tau_values,
            predicted_matrix[index, :],
            linewidth=1.3,
            alpha=0.9,
            label=f"{time_values[index]:g} min fitted",
        )

    if np.all(tau_values > 0):
        ax.set_xscale("log")

    ax.set_title(f"{model_name}: G(tau) profiles observed vs fitted")
    ax.set_xlabel("tau / ms [log]")
    ax.set_ylabel("G(tau)")

    if len(indices) <= 4:
        ax.legend(fontsize="x-small")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_timecourse_at_selected_tau(
    *,
    observed_matrix: np.ndarray,
    predicted_matrix: np.ndarray,
    time_values: np.ndarray,
    tau_values: np.ndarray,
    model_name: str,
    output_path: Path,
    n_tau: int,
):
    indices = np.linspace(0, len(tau_values) - 1, min(n_tau, len(tau_values)))
    indices = sorted(set(int(round(index)) for index in indices))

    fig, ax = plt.subplots(figsize=(8, 5))

    for index in indices:
        ax.plot(
            time_values,
            observed_matrix[:, index],
            marker="o",
            markersize=2,
            linewidth=0.8,
            alpha=0.55,
            label=f"tau={tau_values[index]:.3g} ms observed",
        )

        ax.plot(
            time_values,
            predicted_matrix[:, index],
            linewidth=1.3,
            alpha=0.9,
            label=f"tau={tau_values[index]:.3g} ms fitted",
        )

    ax.set_title(f"{model_name}: timecourses at selected tau values")
    ax.set_xlabel("time_min")
    ax.set_ylabel("G(tau, time)")

    if len(indices) <= 4:
        ax.legend(fontsize="x-small")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot raw FCS surface fit outputs."
    )

    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--figures-dir", default=None)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--sort-by", default="bic")
    parser.add_argument("--n-profiles", type=int, default=6)
    parser.add_argument("--n-tau-timecourses", type=int, default=6)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    figures_dir = Path(args.figures_dir) if args.figures_dir else output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    run_config = read_run_config(output_dir)
    comparison_table = read_comparison_table(output_dir)

    data_path = Path(run_config["data"])
    time_column = run_config.get("time_column", "time_min")

    observed_dataframe = pd.read_csv(data_path)
    time_values, tau_values, observed_matrix = get_surface_matrix(
        observed_dataframe,
        time_column=time_column,
    )

    top_models = choose_top_models(
        comparison_table,
        top_n=args.top_n,
        sort_by=args.sort_by,
    )

    written = []

    written.extend(
        plot_model_weights(
            comparison_table,
            figures_dir,
        )
    )

    plot_surface_heatmap(
        matrix=observed_matrix,
        time_values=time_values,
        tau_values=tau_values,
        output_path=figures_dir / "observed_surface_heatmap.png",
        title="Observed raw FCS surface G(tau, time)",
        label="observed G",
    )
    written.append(figures_dir / "observed_surface_heatmap.png")

    for model_name in top_models:
        model_figures_dir = figures_dir / safe_name(model_name)
        model_figures_dir.mkdir(parents=True, exist_ok=True)

        _, predicted, residuals, summary, _ = load_model_surface(
            output_dir,
            model_name,
        )

        pred_time, pred_tau, predicted_matrix = get_surface_matrix(
            predicted,
            time_column="time_min",
        )

        resid_time, resid_tau, residual_matrix = get_surface_matrix(
            residuals,
            time_column="time_min",
        )

        plot_surface_heatmap(
            matrix=predicted_matrix,
            time_values=pred_time,
            tau_values=pred_tau,
            output_path=model_figures_dir / "predicted_surface_heatmap.png",
            title=f"{model_name}: predicted FCS surface",
            label="predicted G",
        )
        written.append(model_figures_dir / "predicted_surface_heatmap.png")

        plot_surface_heatmap(
            matrix=residual_matrix,
            time_values=resid_time,
            tau_values=resid_tau,
            output_path=model_figures_dir / "residual_surface_heatmap.png",
            title=f"{model_name}: residual FCS surface",
            label="observed - predicted",
        )
        written.append(model_figures_dir / "residual_surface_heatmap.png")

        plot_profiles_at_selected_times(
            observed_matrix=observed_matrix,
            predicted_matrix=predicted_matrix,
            time_values=time_values,
            tau_values=tau_values,
            model_name=model_name,
            output_path=model_figures_dir / "profiles_observed_vs_fitted.png",
            n_profiles=args.n_profiles,
        )
        written.append(model_figures_dir / "profiles_observed_vs_fitted.png")

        plot_timecourse_at_selected_tau(
            observed_matrix=observed_matrix,
            predicted_matrix=predicted_matrix,
            time_values=time_values,
            tau_values=tau_values,
            model_name=model_name,
            output_path=model_figures_dir / "timecourses_at_selected_tau.png",
            n_tau=args.n_tau_timecourses,
        )
        written.append(model_figures_dir / "timecourses_at_selected_tau.png")

    index_path = figures_dir / "surface_figure_index.txt"

    with index_path.open("w") as handle:
        handle.write("FCS surface fit figures\n")
        handle.write("=======================\n\n")
        handle.write(f"Top models: {', '.join(top_models)}\n\n")

        for path in sorted(written):
            handle.write(f"- {path.relative_to(figures_dir)}\n")

    written.append(index_path)

    print("\nFCS surface figures complete")
    print("============================")
    print(f"Figures dir: {figures_dir}")
    print(f"Top models: {', '.join(top_models)}")
    print(f"Figure index: {index_path}")


if __name__ == "__main__":
    main()
