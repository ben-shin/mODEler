from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning

from odefit.engines.registry import get_engine_bundle
from odefit.fitting.variable_projection import project_observables_onto_species


def make_data(
    *,
    n_timepoints: int,
    n_observables: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, list[str]]:
    timepoints = np.linspace(0.0, 10.0, n_timepoints)
    species_values = np.exp(-0.4 * timepoints)

    data = {"time": timepoints}
    signal_columns = []

    for index in range(n_observables):
        column = f"peak_{index}"
        signal_columns.append(column)
        data[column] = (1.0 + 0.001 * index) * species_values + 0.01

    dataframe = pd.DataFrame(data)

    return timepoints, species_values, dataframe, signal_columns


def run_benchmark(
    *,
    n_timepoints: int,
    n_observables: int,
    n_repeats: int,
    engine_name: str,
) -> dict:
    timepoints, species_values, dataframe, signal_columns = make_data(
        n_timepoints=n_timepoints,
        n_observables=n_observables,
    )

    engine_bundle = get_engine_bundle(engine_name)

    # Warmup
    project_observables_onto_species(
        timepoints=timepoints,
        simulated_species_values=species_values,
        observed_dataframe=dataframe,
        signal_columns=signal_columns,
        fit_scale=True,
        fit_offset=True,
        engine_bundle=engine_bundle,
    )

    times = []
    warning_count = 0

    for _ in range(n_repeats):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")

            start = time.perf_counter()

            result = project_observables_onto_species(
                timepoints=timepoints,
                simulated_species_values=species_values,
                observed_dataframe=dataframe,
                signal_columns=signal_columns,
                fit_scale=True,
                fit_offset=True,
                engine_bundle=engine_bundle,
            )

            end = time.perf_counter()

        warning_count += sum(
            1
            for warning in caught
            if issubclass(warning.category, PerformanceWarning)
        )

        times.append(end - start)

    return {
        "benchmark": "variable_projection_dataframe_assembly",
        "engine_name": engine_name,
        "n_timepoints": n_timepoints,
        "n_observables": n_observables,
        "n_repeats": n_repeats,
        "min_seconds": float(np.min(times)),
        "mean_seconds": float(np.mean(times)),
        "median_seconds": float(np.median(times)),
        "max_seconds": float(np.max(times)),
        "performance_warning_count": int(warning_count),
        "rss": float(result.rss),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark variable-projection DataFrame assembly."
    )

    parser.add_argument("--engine-name", default="reference")
    parser.add_argument("--n-timepoints", type=int, default=50)
    parser.add_argument("--n-observables", type=int, default=1000)
    parser.add_argument("--n-repeats", type=int, default=50)
    parser.add_argument(
        "--output-dir",
        default="benchmarks/variable_projection_dataframe_assembly",
    )

    args = parser.parse_args()

    row = run_benchmark(
        n_timepoints=args.n_timepoints,
        n_observables=args.n_observables,
        n_repeats=args.n_repeats,
        engine_name=args.engine_name,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "variable_projection_dataframe_assembly.json"
    csv_path = output_dir / "variable_projection_dataframe_assembly.csv"

    with json_path.open("w") as handle:
        json.dump([row], handle, indent=2)

    pd.DataFrame([row]).to_csv(csv_path, index=False)

    print("\nVariable projection DataFrame assembly benchmark")
    print("================================================")
    print(
        f"{row['engine_name']}: "
        f"median={row['median_seconds']:.6f}s "
        f"mean={row['mean_seconds']:.6f}s "
        f"warnings={row['performance_warning_count']}"
    )

    print("\nWritten files:")
    print(f"  json: {json_path}")
    print(f"  csv: {csv_path}")


if __name__ == "__main__":
    main()
