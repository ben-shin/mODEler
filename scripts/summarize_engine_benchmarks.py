from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_INPUTS = [
    Path("benchmarks/projection_engines/projection_engine_benchmarks.json"),
    Path("benchmarks/batched_projection/batched_projection_benchmarks.json"),
    Path(
        "benchmarks/projection_engine_batch_methods/"
        "projection_engine_batch_method_benchmarks.json"
    ),
    Path(
        "benchmarks/variable_projection_engines/"
        "variable_projection_engine_benchmarks.json"
    ),
]

def dataframe_to_markdown_table(dataframe: pd.DataFrame) -> str:
    """
    Render a small DataFrame as a GitHub-style markdown table without requiring
    pandas' optional tabulate dependency.
    """

    if dataframe.empty:
        return ""

    columns = [str(column) for column in dataframe.columns]

    def format_value(value):
        if pd.isna(value):
            return ""

        if isinstance(value, float):
            return f"{value:.6g}"

        return str(value)

    rows = []

    rows.append("| " + " | ".join(columns) + " |")
    rows.append("| " + " | ".join(["---"] * len(columns)) + " |")

    for _, row in dataframe.iterrows():
        rows.append(
            "| "
            + " | ".join(
                format_value(row[column])
                for column in dataframe.columns
            )
            + " |"
        )

    return "\n".join(rows)

def _load_json_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    data = json.loads(path.read_text())

    if isinstance(data, list):
        return [
            row
            for row in data
            if isinstance(row, dict)
        ]

    if isinstance(data, dict):
        return [data]

    return []


def _normalise_row(row: dict[str, Any], source_path: Path) -> dict[str, Any]:
    benchmark = str(row.get("benchmark", row.get("name", "unknown")))

    engine_name = row.get("engine_name")

    if engine_name is None:
        engine_name = row.get("name", "unknown")

    available = row.get("available", True)

    normalised = {
        "source_file": str(source_path),
        "benchmark": benchmark,
        "engine_name": str(engine_name),
        "available": bool(available),
        "n_timepoints": row.get("n_timepoints"),
        "n_observables": row.get("n_observables"),
        "n_repeats": row.get("n_repeats"),
        "median_seconds": row.get("median_seconds"),
        "mean_seconds": row.get("mean_seconds"),
        "min_seconds": row.get("min_seconds"),
        "max_seconds": row.get("max_seconds"),
        "stdev_seconds": row.get("stdev_seconds"),
        "mean_rss": row.get("mean_rss"),
        "median_rss": row.get("median_rss"),
        "rss": row.get("rss"),
        "mean_nfev": row.get("mean_nfev"),
        "median_k1f": row.get("median_k1f"),
        "error_type": row.get("error_type"),
        "error_message": row.get("error_message"),
    }

    metadata = row.get("metadata")

    if isinstance(metadata, dict):
        for key, value in metadata.items():
            normalised[f"metadata_{key}"] = value

    return normalised


def load_benchmark_rows(paths: list[Path]) -> pd.DataFrame:
    rows = []

    for path in paths:
        for row in _load_json_rows(path):
            rows.append(
                _normalise_row(
                    row,
                    source_path=path,
                )
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "source_file",
                "benchmark",
                "engine_name",
                "available",
                "n_timepoints",
                "n_observables",
                "n_repeats",
                "median_seconds",
                "mean_seconds",
                "min_seconds",
                "max_seconds",
                "stdev_seconds",
                "mean_rss",
                "median_rss",
                "rss",
                "mean_nfev",
                "median_k1f",
                "error_type",
                "error_message",
            ]
        )

    dataframe = pd.DataFrame(rows)

    for column in [
        "n_timepoints",
        "n_observables",
        "n_repeats",
        "median_seconds",
        "mean_seconds",
        "min_seconds",
        "max_seconds",
        "stdev_seconds",
        "mean_rss",
        "median_rss",
        "rss",
        "mean_nfev",
        "median_k1f",
    ]:
        if column in dataframe.columns:
            dataframe[column] = pd.to_numeric(
                dataframe[column],
                errors="coerce",
            )

    return dataframe


def add_speedup_columns(
    dataframe: pd.DataFrame,
    *,
    baseline_engine: str = "reference",
) -> pd.DataFrame:
    if dataframe.empty:
        dataframe["speedup_vs_reference_median"] = pd.Series(dtype=float)
        dataframe["speedup_vs_reference_mean"] = pd.Series(dtype=float)
        return dataframe

    output = dataframe.copy()

    output["speedup_vs_reference_median"] = pd.NA
    output["speedup_vs_reference_mean"] = pd.NA

    grouping_columns = [
        "benchmark",
        "n_timepoints",
        "n_observables",
    ]

    for _, group in output.groupby(grouping_columns, dropna=False):
        baseline_rows = group[
            (group["engine_name"] == baseline_engine)
            & (group["available"])
        ]

        if baseline_rows.empty:
            continue

        baseline_median = baseline_rows.iloc[0].get("median_seconds")
        baseline_mean = baseline_rows.iloc[0].get("mean_seconds")

        if pd.notna(baseline_median) and baseline_median > 0:
            output.loc[group.index, "speedup_vs_reference_median"] = (
                baseline_median / group["median_seconds"]
            )

        if pd.notna(baseline_mean) and baseline_mean > 0:
            output.loc[group.index, "speedup_vs_reference_mean"] = (
                baseline_mean / group["mean_seconds"]
            )

    return output


def make_markdown_summary(
    dataframe: pd.DataFrame,
    *,
    baseline_engine: str = "reference",
) -> str:
    lines = [
        "# Engine Benchmark Summary",
        "",
        f"Baseline engine: `{baseline_engine}`",
        "",
    ]

    if dataframe.empty:
        lines.extend(
            [
                "No benchmark rows were found.",
                "",
            ]
        )
        return "\n".join(lines)

    available = dataframe[dataframe["available"]].copy()
    unavailable = dataframe[~dataframe["available"]].copy()

    lines.extend(
        [
            "## Available benchmark rows",
            "",
        ]
    )

    if available.empty:
        lines.append("No available benchmark rows.")
        lines.append("")
    else:
        display_columns = [
            "benchmark",
            "engine_name",
            "n_timepoints",
            "n_observables",
            "n_repeats",
            "median_seconds",
            "mean_seconds",
            "speedup_vs_reference_median",
            "mean_nfev",
            "median_k1f",
            "median_rss",
            "rss",
        ]

        existing_columns = [
            column
            for column in display_columns
            if column in available.columns
        ]

        table = available[existing_columns].sort_values(
            by=[
                "benchmark",
                "n_timepoints",
                "n_observables",
                "engine_name",
            ],
            na_position="last",
        )

        lines.append(
            dataframe_to_markdown_table(table)
        )
        lines.append("")

    lines.extend(
        [
            "## Unavailable engines",
            "",
        ]
    )

    if unavailable.empty:
        lines.append("No unavailable engines were reported.")
        lines.append("")
    else:
        display_columns = [
            "benchmark",
            "engine_name",
            "error_type",
            "error_message",
        ]

        existing_columns = [
            column
            for column in display_columns
            if column in unavailable.columns
        ]

        lines.append(
            dataframe_to_markdown_table(unavailable[existing_columns])
            )
        lines.append("")

    lines.extend(
        [
            "## Notes",
            "",
            "- Speedup values are computed as reference median time divided by engine median time.",
            "- Values greater than 1 mean faster than reference.",
            "- Workflow-level benchmarks include API/config overhead, ODE solving, optimizer calls, and projection.",
            "- Kernel-level benchmarks isolate smaller pieces and may exaggerate apparent speedups.",
            "",
        ]
    )

    return "\n".join(lines)


def write_summary_outputs(
    dataframe: pd.DataFrame,
    *,
    output_dir: Path,
    baseline_engine: str,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "engine_benchmark_summary.csv"
    markdown_path = output_dir / "engine_benchmark_summary.md"
    json_path = output_dir / "engine_benchmark_summary.json"

    dataframe.to_csv(csv_path, index=False)

    markdown_path.write_text(
        make_markdown_summary(
            dataframe,
            baseline_engine=baseline_engine,
        )
    )

    payload = dataframe.where(
        pd.notna(dataframe),
        None,
    ).to_dict(orient="records")

    with json_path.open("w") as handle:
        json.dump(payload, handle, indent=2)

    return {
        "csv": csv_path,
        "markdown": markdown_path,
        "json": json_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize engine benchmark outputs."
    )

    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[str(path) for path in DEFAULT_INPUTS],
        help="Benchmark JSON files to summarize.",
    )

    parser.add_argument(
        "--output-dir",
        default="benchmarks/summary",
    )

    parser.add_argument(
        "--baseline-engine",
        default="reference",
    )

    args = parser.parse_args()

    paths = [
        Path(path)
        for path in args.inputs
    ]

    dataframe = load_benchmark_rows(paths)

    dataframe = add_speedup_columns(
        dataframe,
        baseline_engine=args.baseline_engine,
    )

    written_files = write_summary_outputs(
        dataframe,
        output_dir=Path(args.output_dir),
        baseline_engine=args.baseline_engine,
    )

    print("\nEngine benchmark summary")
    print("========================")
    print(f"Rows: {len(dataframe)}")

    if dataframe.empty:
        print("No benchmark data found.")
    else:
        available_count = int(dataframe["available"].sum())
        unavailable_count = int((~dataframe["available"]).sum())

        print(f"Available rows: {available_count}")
        print(f"Unavailable rows: {unavailable_count}")

    print("\nWritten files:")
    for name, path in written_files.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
