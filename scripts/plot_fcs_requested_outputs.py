from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def safe_model_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_")


def load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def find_time_column(dataframe: pd.DataFrame, preferred: str | None = None) -> str:
    if preferred and preferred in dataframe.columns:
        return preferred

    for candidate in ["time_min", "tau", "time", "lag_time", "lag", "t"]:
        if candidate in dataframe.columns:
            return candidate

    return dataframe.columns[0]


def get_signal_columns(
    dataframe: pd.DataFrame,
    *,
    time_column: str,
    requested: list[str] | None,
) -> list[str]:
    if requested:
        return [column for column in requested if column in dataframe.columns]

    numeric_columns = dataframe.select_dtypes(include="number").columns.tolist()

    return [
        column
        for column in numeric_columns
        if column != time_column
    ]


def read_generated_config(output_dir: Path) -> dict:
    config_path = output_dir / "generated_fcs_all_txt_config.json"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Could not find generated config: {config_path}. "
            "Rerun scripts/run_fcs_txt_models.py first."
        )

    return load_json(config_path)


def read_comparison_table(output_dir: Path) -> pd.DataFrame:
    table_path = output_dir / "fcs_all_txt_model_comparison_table.csv"

    if not table_path.exists():
        raise FileNotFoundError(
            f"Could not find comparison table: {table_path}. "
            "Rerun scripts/run_fcs_txt_models.py first."
        )

    return pd.read_csv(table_path)


def compute_information_criterion_weights(
    comparison_table: pd.DataFrame,
    *,
    criterion: str,
) -> pd.DataFrame:
    if criterion not in comparison_table.columns:
        raise ValueError(
            f"Cannot compute {criterion} weights because column {criterion!r} "
            "is missing from the comparison table."
        )

    table = comparison_table.copy()
    table = table[pd.to_numeric(table[criterion], errors="coerce").notna()].copy()

    if "success" in table.columns:
        table = table[table["success"].astype(bool)].copy()

    if table.empty:
        raise ValueError(f"No successful rows with numeric {criterion} values.")

    values = table[criterion].astype(float)
    delta = values - values.min()

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


def write_model_weight_outputs(
    *,
    comparison_table: pd.DataFrame,
    figures_dir: Path,
    output_dir: Path,
    criteria: list[str],
) -> None:
    weight_tables = []

    for criterion in criteria:
        if criterion not in comparison_table.columns:
            continue

        weights = compute_information_criterion_weights(
            comparison_table,
            criterion=criterion,
        )

        weights_path = output_dir / f"model_{criterion}_weights.csv"
        weights.to_csv(weights_path, index=False)

        weight_tables.append((criterion, weights))

        fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(weights)), 5))

        ax.bar(
            weights["model_name"].astype(str),
            weights[f"{criterion}_weight_percent"].astype(float),
        )

        ax.set_title(f"Relative model weights from {criterion.upper()}")
        ax.set_xlabel("model")
        ax.set_ylabel("relative model weight (%)")
        ax.tick_params(axis="x", rotation=45)

        fig.tight_layout()
        fig.savefig(figures_dir / f"model_{criterion}_weight_percent.png", dpi=200)
        plt.close(fig)

    if weight_tables:
        summary = comparison_table.copy()

        for criterion, weights in weight_tables:
            keep_columns = [
                "model_name",
                f"delta_{criterion}",
                f"{criterion}_weight",
                f"{criterion}_weight_percent",
            ]

            summary = summary.merge(
                weights[keep_columns],
                on="model_name",
                how="left",
            )

        summary.to_csv(output_dir / "model_weight_summary.csv", index=False)


def load_model_outputs(output_dir: Path, model_name: str):
    model_dir = output_dir / "models" / safe_model_name(model_name)

    predicted_path = model_dir / "predicted.csv"
    residuals_path = model_dir / "residuals.csv"
    observable_table_path = model_dir / "observable_table.csv"

    if not predicted_path.exists():
        raise FileNotFoundError(
            f"Missing {predicted_path}. "
            "Patch/rerun scripts/run_fcs_txt_models.py so it exports per-model outputs."
        )

    predicted = pd.read_csv(predicted_path)

    residuals = (
        pd.read_csv(residuals_path)
        if residuals_path.exists()
        else pd.DataFrame()
    )

    observable_table = (
        pd.read_csv(observable_table_path)
        if observable_table_path.exists()
        else pd.DataFrame()
    )

    return model_dir, predicted, residuals, observable_table


def choose_models_to_plot(
    comparison_table: pd.DataFrame,
    *,
    top_n: int,
    sort_by: str,
) -> list[str]:
    table = comparison_table.copy()

    if "success" in table.columns:
        table = table[table["success"].astype(bool)].copy()

    if sort_by in table.columns:
        table = table.sort_values(sort_by, na_position="last")

    if table.empty:
        raise ValueError("No successful models available for plotting.")

    return [
        str(model_name)
        for model_name in table["model_name"].head(top_n)
    ]


def plot_each_timecourse_for_model(
    *,
    observed: pd.DataFrame,
    predicted: pd.DataFrame,
    model_name: str,
    time_column: str,
    signal_columns: list[str],
    output_dir: Path,
    max_individual_plots: int | None,
) -> None:
    model_plot_dir = output_dir / "individual_timecourses" / safe_model_name(model_name)
    model_plot_dir.mkdir(parents=True, exist_ok=True)

    observed_time = observed[time_column].to_numpy(dtype=float)

    predicted_time_column = find_time_column(predicted, preferred=time_column)
    predicted_time = predicted[predicted_time_column].to_numpy(dtype=float)

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

        ax.set_title(f"{model_name}: {column}")
        ax.set_xlabel(time_column)
        ax.set_ylabel("signal")
        ax.legend()

        fig.tight_layout()
        fig.savefig(model_plot_dir / f"{column}.png", dpi=200)
        plt.close(fig)


def plot_combined_timecourses_for_model(
    *,
    observed: pd.DataFrame,
    predicted: pd.DataFrame,
    model_name: str,
    time_column: str,
    signal_columns: list[str],
    output_path: Path,
    max_traces: int | None,
) -> None:
    observed_time = observed[time_column].to_numpy(dtype=float)

    predicted_time_column = find_time_column(predicted, preferred=time_column)
    predicted_time = predicted[predicted_time_column].to_numpy(dtype=float)

    columns_to_plot = signal_columns

    if max_traces is not None and len(columns_to_plot) > max_traces:
        indices = np.linspace(0, len(columns_to_plot) - 1, max_traces)
        indices = sorted(set(int(round(index)) for index in indices))
        columns_to_plot = [columns_to_plot[index] for index in indices]

    fig, ax = plt.subplots(figsize=(10, 6))

    for column in columns_to_plot:
        if column not in observed.columns or column not in predicted.columns:
            continue

        ax.plot(
            observed_time,
            observed[column].to_numpy(dtype=float),
            linewidth=0.7,
            alpha=0.30,
        )

    for column in columns_to_plot:
        if column not in predicted.columns:
            continue

        ax.plot(
            predicted_time,
            predicted[column].to_numpy(dtype=float),
            linewidth=0.9,
            alpha=0.80,
        )

    ax.set_title(f"{model_name}: all selected timecourses observed + fitted")
    ax.set_xlabel(time_column)
    ax.set_ylabel("signal")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_combined_all_models_best_fits(
    *,
    observed: pd.DataFrame,
    output_dir: Path,
    model_names: list[str],
    time_column: str,
    signal_columns: list[str],
    figures_dir: Path,
    max_traces: int,
) -> None:
    observed_time = observed[time_column].to_numpy(dtype=float)

    selected_columns = signal_columns

    if len(selected_columns) > max_traces:
        indices = np.linspace(0, len(selected_columns) - 1, max_traces)
        indices = sorted(set(int(round(index)) for index in indices))
        selected_columns = [selected_columns[index] for index in indices]

    for model_name in model_names:
        _, predicted, _, _ = load_model_outputs(output_dir, model_name)

        predicted_time_column = find_time_column(predicted, preferred=time_column)
        predicted_time = predicted[predicted_time_column].to_numpy(dtype=float)

        fig, ax = plt.subplots(figsize=(10, 6))

        for column in selected_columns:
            if column in observed.columns:
                ax.plot(
                    observed_time,
                    observed[column].to_numpy(dtype=float),
                    linewidth=0.7,
                    alpha=0.25,
                )

        for column in selected_columns:
            if column in predicted.columns:
                ax.plot(
                    predicted_time,
                    predicted[column].to_numpy(dtype=float),
                    linewidth=1.0,
                    alpha=0.85,
                )

        ax.set_title(f"{model_name}: combined observed and fitted timecourses")
        ax.set_xlabel(time_column)
        ax.set_ylabel("signal")

        fig.tight_layout()
        fig.savefig(
            figures_dir / f"combined_timecourses_{safe_model_name(model_name)}.png",
            dpi=200,
        )
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate requested FCS figures: per-timecourse fits, combined fits, "
            "and model weight percentages."
        )
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory from scripts/run_fcs_txt_models.py.",
    )

    parser.add_argument(
        "--figures-dir",
        default=None,
        help="Where to write figures. Defaults to <output-dir>/requested_figures.",
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=3,
        help="Number of top-ranked models to plot.",
    )

    parser.add_argument(
        "--sort-by",
        default="bic",
        help="Ranking column used to choose top models.",
    )

    parser.add_argument(
        "--max-individual-plots",
        type=int,
        default=None,
        help=(
            "Maximum individual signal/timecourse plots per model. "
            "Default is all signal columns."
        ),
    )

    parser.add_argument(
        "--max-combined-traces",
        type=int,
        default=80,
        help="Maximum traces shown in combined plots.",
    )

    parser.add_argument(
        "--criteria",
        nargs="+",
        default=["bic", "aic"],
        help="Information criteria for model percentage weights.",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    figures_dir = (
        Path(args.figures_dir)
        if args.figures_dir is not None
        else output_dir / "requested_figures"
    )
    figures_dir.mkdir(parents=True, exist_ok=True)

    config = read_generated_config(output_dir)
    comparison_table = read_comparison_table(output_dir)

    data_path = Path(config["data"])
    observed = pd.read_csv(data_path)

    time_column = find_time_column(
        observed,
        preferred=config.get("time_column"),
    )

    signal_columns = get_signal_columns(
        observed,
        time_column=time_column,
        requested=config.get("signal_columns"),
    )

    model_names = choose_models_to_plot(
        comparison_table,
        top_n=args.top_n,
        sort_by=args.sort_by,
    )

    write_model_weight_outputs(
        comparison_table=comparison_table,
        figures_dir=figures_dir,
        output_dir=output_dir,
        criteria=args.criteria,
    )

    for model_name in model_names:
        _, predicted, _, _ = load_model_outputs(output_dir, model_name)

        plot_each_timecourse_for_model(
            observed=observed,
            predicted=predicted,
            model_name=model_name,
            time_column=time_column,
            signal_columns=signal_columns,
            output_dir=figures_dir,
            max_individual_plots=args.max_individual_plots,
        )

        plot_combined_timecourses_for_model(
            observed=observed,
            predicted=predicted,
            model_name=model_name,
            time_column=time_column,
            signal_columns=signal_columns,
            output_path=(
                figures_dir
                / f"combined_timecourses_{safe_model_name(model_name)}.png"
            ),
            max_traces=args.max_combined_traces,
        )

    plot_combined_all_models_best_fits(
        observed=observed,
        output_dir=output_dir,
        model_names=model_names,
        time_column=time_column,
        signal_columns=signal_columns,
        figures_dir=figures_dir,
        max_traces=args.max_combined_traces,
    )

    index_path = figures_dir / "requested_figure_index.txt"

    with index_path.open("w") as handle:
        handle.write("Requested FCS figures\n")
        handle.write("=====================\n\n")
        handle.write(f"Output dir: {output_dir}\n")
        handle.write(f"Figures dir: {figures_dir}\n")
        handle.write(f"Top models: {', '.join(model_names)}\n")
        handle.write(f"Time column: {time_column}\n")
        handle.write(f"Signal columns: {len(signal_columns)}\n\n")

        handle.write("Model weight files:\n")
        for criterion in args.criteria:
            handle.write(f"- model_{criterion}_weights.csv\n")
            handle.write(f"- model_{criterion}_weight_percent.png\n")

        handle.write("\nFigure files:\n")
        for path in sorted(figures_dir.rglob("*.png")):
            handle.write(f"- {path.relative_to(figures_dir)}\n")

    print("\nRequested FCS figures complete")
    print("==============================")
    print(f"Output dir: {output_dir}")
    print(f"Figures dir: {figures_dir}")
    print(f"Top models plotted: {', '.join(model_names)}")
    print(f"Figure index: {index_path}")
    print(f"Model weight summary: {output_dir / 'model_weight_summary.csv'}")


if __name__ == "__main__":
    main()
