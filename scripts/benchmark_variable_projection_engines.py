from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from odefit.api.backend import fit_global_observables_from_config


def make_variable_projection_data(
    *,
    output_path: Path,
    n_timepoints: int,
    n_observables: int,
    random_seed: int,
) -> Path:
    rng = np.random.default_rng(random_seed)

    timepoints = np.linspace(0.0, 10.0, n_timepoints)
    species_a = np.exp(-0.4 * timepoints)

    scales = rng.uniform(0.5, 2.0, size=n_observables)
    offsets = rng.uniform(-0.1, 0.1, size=n_observables)

    data = {"time": timepoints}

    for index in range(n_observables):
        data[f"peak_{index}"] = (
            scales[index] * species_a
            + offsets[index]
            + rng.normal(0.0, 0.002, size=n_timepoints)
        )

    dataframe = pd.DataFrame(data)
    dataframe.to_csv(output_path, index=False)

    return output_path


def make_fit_config(
    *,
    data_path: Path,
    engine_name: str,
    max_nfev: int,
) -> dict:
    return {
        "engine_name": engine_name,
        "model_text": "A -> B",
        "data": str(data_path),
        "time_column": "time",
        "observed_species": "A",
        "use_variable_projection": True,
        "fit_scale": True,
        "fit_offset": True,
        "parameters": {
            "k1f": {
                "initial_guess": 0.2,
                "lower_bound": 1e-6,
                "upper_bound": 10.0,
            }
        },
        "initial_conditions": {
            "A": {
                "value": 1.0,
                "mode": "fixed",
            },
            "B": {
                "value": 0.0,
                "mode": "fixed",
            },
        },
        "method": "trf",
        "loss": "linear",
        "max_nfev": max_nfev,
        "rtol": 1e-6,
        "atol": 1e-9,
        "variable_projection_backend": "numpy",
        "variable_projection_method": "LSODA",
        "show_progress": False,
        "no_plots": True,
    }


def benchmark_engine(
    *,
    engine_name: str,
    data_path: Path,
    n_repeats: int,
    max_nfev: int,
) -> dict:
    config = make_fit_config(
        data_path=data_path,
        engine_name=engine_name,
        max_nfev=max_nfev,
    )

    # Warmup. Important for engines with JIT compilation.
    warmup_output = fit_global_observables_from_config(config)
    warmup_result = warmup_output["result"]

    times = []
    fitted_k_values = []
    rss_values = []
    nfev_values = []

    for _ in range(n_repeats):
        start = time.perf_counter()
        output = fit_global_observables_from_config(config)
        end = time.perf_counter()

        result = output["result"]

        times.append(end - start)
        fitted_k_values.append(float(result.fitted_parameters["k1f"]))
        rss_values.append(float(result.statistics["rss"]))
        nfev_values.append(int(result.nfev))

    return {
        "engine_name": engine_name,
        "available": True,
        "benchmark": "single_species_variable_projection_fit",
        "n_repeats": int(n_repeats),
        "max_nfev": int(max_nfev),
        "min_seconds": float(np.min(times)),
        "mean_seconds": float(np.mean(times)),
        "median_seconds": float(np.median(times)),
        "max_seconds": float(np.max(times)),
        "mean_k1f": float(np.mean(fitted_k_values)),
        "median_k1f": float(np.median(fitted_k_values)),
        "mean_rss": float(np.mean(rss_values)),
        "median_rss": float(np.median(rss_values)),
        "mean_nfev": float(np.mean(nfev_values)),
        "warmup_success": bool(warmup_result.success),
        "warmup_k1f": float(warmup_result.fitted_parameters["k1f"]),
        "warmup_rss": float(warmup_result.statistics["rss"]),
        "times_seconds": times,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark full variable-projection workflows by engine."
    )

    parser.add_argument(
        "--engines",
        nargs="+",
        default=["reference", "numba_projection", "jax_projection"],
    )

    parser.add_argument(
        "--n-timepoints",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--n-observables",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--n-repeats",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--max-nfev",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=123,
    )

    parser.add_argument(
        "--output-dir",
        default="benchmarks/variable_projection_engines",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_path = output_dir / "synthetic_variable_projection_data.csv"

    make_variable_projection_data(
        output_path=data_path,
        n_timepoints=args.n_timepoints,
        n_observables=args.n_observables,
        random_seed=args.random_seed,
    )

    rows = []

    for engine_name in args.engines:
        try:
            row = benchmark_engine(
                engine_name=engine_name,
                data_path=data_path,
                n_repeats=args.n_repeats,
                max_nfev=args.max_nfev,
            )

            print(
                f"{engine_name}: "
                f"median={row['median_seconds']:.6f}s "
                f"mean={row['mean_seconds']:.6f}s "
                f"k1f={row['median_k1f']:.6g} "
                f"rss={row['median_rss']:.6g} "
                f"nfev={row['mean_nfev']:.1f}"
            )

        except Exception as exc:
            row = {
                "engine_name": engine_name,
                "available": False,
                "benchmark": "single_species_variable_projection_fit",
                "n_repeats": int(args.n_repeats),
                "max_nfev": int(args.max_nfev),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }

            print(
                f"{engine_name}: unavailable "
                f"({type(exc).__name__}: {exc})"
            )

        rows.append(row)

    json_path = output_dir / "variable_projection_engine_benchmarks.json"
    csv_path = output_dir / "variable_projection_engine_benchmarks.csv"

    with json_path.open("w") as handle:
        json.dump(rows, handle, indent=2)

    flattened_rows = []

    for row in rows:
        flattened = {
            key: value
            for key, value in row.items()
            if key != "times_seconds"
        }
        flattened_rows.append(flattened)

    pd.DataFrame(flattened_rows).to_csv(csv_path, index=False)

    print("\nWrote data:", data_path)
    print("Wrote JSON:", json_path)
    print("Wrote CSV:", csv_path)


if __name__ == "__main__":
    main()
