from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _safe_model_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_")


def _load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def _find_time_column(dataframe: pd.DataFrame, preferred: str | None = None) -> str:
    if preferred and preferred in dataframe.columns:
        return preferred

    for candidate in ["time_min", "tau", "time", "lag_time", "t"]:
        if candidate in dataframe.columns:
            return candidate

    return dataframe.columns[0]


def _numeric_signal_columns(
    dataframe: pd.DataFrame,
    *,
    time_column: str,
    requested: list[str] | None = None,
) -> list[str]:
    if requested:
        return [column for column in requested if column in dataframe.columns]

    numeric_columns = dataframe.select_dtypes(include="number").columns.tolist()

    return [column for column in numeric_columns if column != time_column]


def _read_config(output_dir: Path) -> dict:
    path = output_dir / "generated_fcs_all_txt_config.json"

    if not path.exists():
        return {}

    return _load_json(path)


def _read_comparison_table(output_dir: Path) -> pd.DataFrame:
    path = output_dir / "fcs_all_txt_model_comparison_table.csv"

    if not path.exists():
        raise FileNotFoundError(f"Missing comparison table: {path}")

    return pd.read_csv(path)


def _read_model_summary(model_dir: Path) -> dict:
    path = model_dir / "fit_summary.json"

    if not path.exists():
        return {}

    return _load_json(path)


def _read_model_data(output_dir: Path, model_name: str):
    model_dir = output_dir / "models" / _safe_model_name(model_name)

    predicted_path = model_dir / "predicted.csv"
    residuals_path = model_dir / "residuals.csv"
    observable_path = model_dir / "observable_table.csv"

    if not predicted_path.exists():
        raise FileNotFoundError(f"Missing predicted CSV: {predicted_path}")

    if not residuals_path.exists():
        raise FileNotFoundError(f"Missing residuals CSV: {residuals_path}")

    predicted = pd.read_csv(predicted_path)
    residuals = pd.read_csv(residuals_path)

    observable = (
        pd.read_csv(observable_path)
        if observable_path.exists()
        else pd.DataFrame()
    )

    summary = _read_model_summary(model_dir)

    return model_dir, predicted, residuals, observable, summary


def _save_bar_plot(
    *,
    dataframe: pd.DataFrame,
    x_column: str,
    y_column: str,
    output_path: Path,
    title: str,
    ylabel: str,
    max_items: int | None = None,
):
    if y_column not in dataframe.columns:
        return

    plot_df = dataframe.dropna(subset=[y_column]).copy()

    if plot_df.empty:
        return

    plot_df = plot_df.sort_values(y_column)

    if max_items is not None:
        plot_df = plot_df.head(max_items)

    fig, ax = plt.subplots(figsize=(max(8, 0.5 * len(plot_df)), 5))

    ax.bar(plot_df[x_column].astype(str), plot_df[y_column].astype(float))
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("model")
    ax.tick_params(axis="x", rotation=45)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_model_rankings(
    *,
    comparison_table: pd.DataFrame,
    figures_dir: Path,
):
    _save_bar_plot(
        dataframe=comparison_table,
        x_column="model_name",
        y_column="bic",
        output_path=figures_dir / "model_ranking_bic.png",
        title="Model ranking by BIC",
        ylabel="BIC",
    )

    _save_bar_plot(
        dataframe=comparison_table,
        x_column="model_name",
        y_column="aic",
        output_path=figures_dir / "model_ranking_aic.png",
        title="Model ranking by AIC",
        ylabel="AIC",
    )

    _save_bar_plot(
        dataframe=comparison_table,
        x_column="model_name",
        y_column="rss",
        output_path=figures_dir / "model_ranking_rss.png",
        title="Model ranking by RSS",
        ylabel="RSS",
    )


def plot_top_models_bic(
    *,
    comparison_table: pd.DataFrame,
    figures_dir: Path,
    top_n: int,
):
    if "bic" not in comparison_table.columns:
        return

    _save_bar_plot(
        dataframe=comparison_table,
        x_column="model_name",
        y_column="bic",
        output_path=figures_dir / "top_models_bic_comparison.png",
        title=f"Top {top_n} models by BIC",
        ylabel="BIC",
        max_items=top_n,
    )


def _selected_columns(
    signal_columns: list[str],
    *,
    max_traces: int,
) -> list[str]:
    if len(signal_columns) <= max_traces:
        return signal_columns

    indices = np.linspace(0, len(signal_columns) - 1, max_traces)
    indices = sorted(set(int(round(index)) for index in indices))

    return [signal_columns[index] for index in indices]


def plot_overlay_all_traces(
    *,
    observed: pd.DataFrame,
    predicted: pd.DataFrame,
    time_column: str,
    signal_columns: list[str],
    output_path: Path,
    title: str,
):
    fig, ax = plt.subplots(figsize=(9, 6))

    x = observed[time_column].to_numpy(dtype=float)

    for column in signal_columns:
        if column in observed.columns:
            ax.plot(
                x,
                observed[column].to_numpy(dtype=float),
                linewidth=0.7,
                alpha=0.25,
            )

    pred_time = _find_time_column(predicted, preferred=time_column)
    xp = predicted[pred_time].to_numpy(dtype=float)

    for column in signal_columns:
        if column in predicted.columns:
            ax.plot(
                xp,
                predicted[column].to_numpy(dtype=float),
                linewidth=0.7,
                alpha=0.7,
            )

    ax.set_title(title)
    ax.set_xlabel(time_column)
    ax.set_ylabel("signal")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_overlay_selected_traces(
    *,
    observed: pd.DataFrame,
    predicted: pd.DataFrame,
    time_column: str,
    signal_columns: list[str],
    output_path: Path,
    title: str,
):
    selected = _selected_columns(signal_columns, max_traces=12)

    fig, ax = plt.subplots(figsize=(10, 6))

    x = observed[time_column].to_numpy(dtype=float)
    pred_time = _find_time_column(predicted, preferred=time_column)
    xp = predicted[pred_time].to_numpy(dtype=float)

    for column in selected:
        if column not in observed.columns or column not in predicted.columns:
            continue

        ax.plot(
            x,
            observed[column].to_numpy(dtype=float),
            marker="o",
            markersize=3,
            linewidth=0.8,
            alpha=0.65,
            label=f"{column} observed",
        )

        ax.plot(
            xp,
            predicted[column].to_numpy(dtype=float),
            linewidth=1.2,
            alpha=0.9,
            label=f"{column} fit",
        )

    ax.set_title(title)
    ax.set_xlabel(time_column)
    ax.set_ylabel("signal")

    if len(selected) <= 6:
        ax.legend(fontsize="small")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_residual_heatmap(
    *,
    residuals: pd.DataFrame,
    time_column: str,
    signal_columns: list[str],
    output_path: Path,
    title: str,
):
    residual_time = _find_time_column(residuals, preferred=time_column)
    available = [column for column in signal_columns if column in residuals.columns]

    if not available:
        return

    matrix = residuals[available].to_numpy(dtype=float).T

    fig, ax = plt.subplots(figsize=(10, max(5, 0.03 * len(available))))

    image = ax.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
    )

    ax.set_title(title)
    ax.set_xlabel("time index")
    ax.set_ylabel("signal column index")

    if len(available) <= 30:
        ax.set_yticks(np.arange(len(available)))
        ax.set_yticklabels(available, fontsize=6)

    fig.colorbar(image, ax=ax, label="residual")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_observable_rss(
    *,
    observable: pd.DataFrame,
    output_path: Path,
    title: str,
):
    if observable.empty or "rss" not in observable.columns:
        return

    column_name = "data_column" if "data_column" in observable.columns else "signal_column"

    if column_name not in observable.columns:
        observable = observable.copy()
        observable["signal_column"] = np.arange(len(observable)).astype(str)
        column_name = "signal_column"

    plot_df = observable.sort_values("rss", ascending=False).head(50)

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.bar(plot_df[column_name].astype(str), plot_df["rss"].astype(float))
    ax.set_title(title)
    ax.set_xlabel("signal column")
    ax.set_ylabel("RSS")
    ax.tick_params(axis="x", rotation=90, labelsize=6)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_parameters(
    *,
    summary: dict,
    output_path: Path,
    title: str,
):
    parameters = summary.get("fitted_parameters", {})

    if not parameters:
        return

    names = list(parameters)
    values = [float(parameters[name]) for name in names]

    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(names)), 4))

    ax.bar(names, values)
    ax.set_title(title)
    ax.set_xlabel("parameter")
    ax.set_ylabel("value")
    ax.tick_params(axis="x", rotation=45)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_one_model(
    *,
    output_dir: Path,
    observed: pd.DataFrame,
    time_column: str,
    signal_columns: list[str],
    model_name: str,
    figures_dir: Path,
    best_prefix: str | None = None,
):
    model_dir, predicted, residuals, observable, summary = _read_model_data(
        output_dir,
        model_name,
    )

    model_figures_dir = figures_dir / _safe_model_name(model_name)
    model_figures_dir.mkdir(parents=True, exist_ok=True)

    plot_overlay_selected_traces(
        observed=observed,
        predicted=predicted,
        time_column=time_column,
        signal_columns=signal_columns,
        output_path=model_figures_dir / "overlay_selected_traces.png",
        title=f"{model_name}: selected observed vs fit traces",
    )

    plot_residual_heatmap(
        residuals=residuals,
        time_column=time_column,
        signal_columns=signal_columns,
        output_path=model_figures_dir / "residual_heatmap.png",
        title=f"{model_name}: residual heatmap",
    )

    plot_observable_rss(
        observable=observable,
        output_path=model_figures_dir / "observable_rss.png",
        title=f"{model_name}: worst observable RSS values",
    )

    plot_parameters(
        summary=summary,
        output_path=model_figures_dir / "parameter_values.png",
        title=f"{model_name}: fitted parameters",
    )

    if best_prefix:
        plot_overlay_all_traces(
            observed=observed,
            predicted=predicted,
            time_column=time_column,
            signal_columns=signal_columns,
            output_path=figures_dir / f"{best_prefix}_overlay_all_traces.png",
            title=f"{model_name}: all observed vs fit traces",
        )

        plot_overlay_selected_traces(
            observed=observed,
            predicted=predicted,
            time_column=time_column,
            signal_columns=signal_columns,
            output_path=figures_dir / f"{best_prefix}_overlay_selected_traces.png",
            title=f"{model_name}: selected observed vs fit traces",
        )

        plot_residual_heatmap(
            residuals=residuals,
            time_column=time_column,
            signal_columns=signal_columns,
            output_path=figures_dir / f"{best_prefix}_residual_heatmap.png",
            title=f"{model_name}: residual heatmap",
        )

        plot_observable_rss(
            observable=observable,
            output_path=figures_dir / f"{best_prefix}_observable_rss.png",
            title=f"{model_name}: worst observable RSS values",
        )

        plot_parameters(
            summary=summary,
            output_path=figures_dir / f"{best_prefix}_parameter_values.png",
            title=f"{model_name}: fitted parameters",
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot FCS all-TXT model comparison outputs."
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory from scripts/run_fcs_txt_models.py.",
    )

    parser.add_argument(
        "--figures-dir",
        default=None,
        help="Figure output directory. Defaults to <output-dir>/figures.",
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=3,
        help="Number of top models to make per-model figures for.",
    )

    parser.add_argument(
        "--sort-by",
        default=None,
        help="Ranking column. Defaults to config/output sort_by, then bic.",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    figures_dir = Path(args.figures_dir) if args.figures_dir else output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    config = _read_config(output_dir)
    comparison_table = _read_comparison_table(output_dir)

    data_path = Path(config.get("data", ""))

    if not data_path.exists():
        raise FileNotFoundError(
            "Could not find original data path from generated config: "
            f"{data_path}"
        )

    observed = pd.read_csv(data_path)

    time_column = _find_time_column(
        observed,
        preferred=config.get("time_column"),
    )

    signal_columns = _numeric_signal_columns(
        observed,
        time_column=time_column,
        requested=config.get("signal_columns"),
    )

    sort_by = args.sort_by or config.get("sort_by") or "bic"

    if sort_by in comparison_table.columns:
        ranked = comparison_table.sort_values(sort_by, na_position="last")
    else:
        ranked = comparison_table.copy()

    ranked = ranked[ranked["success"].astype(bool)] if "success" in ranked.columns else ranked

    if ranked.empty:
        raise RuntimeError("No successful models found to plot.")

    best_model = str(ranked.iloc[0]["model_name"])
    top_models = [str(value) for value in ranked["model_name"].head(args.top_n)]

    plot_model_rankings(
        comparison_table=comparison_table,
        figures_dir=figures_dir,
    )

    plot_top_models_bic(
        comparison_table=comparison_table,
        figures_dir=figures_dir,
        top_n=args.top_n,
    )

    for index, model_name in enumerate(top_models):
        plot_one_model(
            output_dir=output_dir,
            observed=observed,
            time_column=time_column,
            signal_columns=signal_columns,
            model_name=model_name,
            figures_dir=figures_dir,
            best_prefix="best_model" if model_name == best_model else None,
        )

    index_path = figures_dir / "figure_index.txt"

    with index_path.open("w") as handle:
        handle.write("FCS model comparison figures\n")
        handle.write("============================\n\n")
        handle.write(f"Best model: {best_model}\n")
        handle.write(f"Top models plotted: {', '.join(top_models)}\n\n")

        for path in sorted(figures_dir.rglob("*.png")):
            handle.write(str(path.relative_to(figures_dir)) + "\n")

    print("\nFCS figures written")
    print("===================")
    print(f"Output dir: {output_dir}")
    print(f"Figures dir: {figures_dir}")
    print(f"Best model: {best_model}")
    print(f"Top models plotted: {', '.join(top_models)}")
    print(f"Figure index: {index_path}")


if __name__ == "__main__":
    main()
