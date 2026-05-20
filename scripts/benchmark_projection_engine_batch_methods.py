from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from odefit.engines.registry import get_engine_bundle


def make_projection_data(
    *,
    n_timepoints: int,
    n_observables: int,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_seed)

    timepoints = np.linspace(0.0, 10.0, n_timepoints)
    species_values = np.exp(-0.4 * timepoints)

    scales = rng.uniform(0.5, 2.0, size=n_observables)
    offsets = rng.uniform(-0.1, 0.1, size=n_observables)

    observed_matrix = (
        species_values[:, None] * scales[None, :]
        + offsets[None, :]
        + rng.normal(0.0, 0.002, size=(n_timepoints, n_observables))
    )

    return species_values, observed_matrix


def time_engine_batch_projection(
    *,
    engine_name: str,
    species_values: np.ndarray,
    observed_matrix: np.ndarray,
    n_repeats: int,
) -> dict:
    engine = get_engine_bundle(engine_name)

    # Warmup. Important for JIT engines.
    warmup_result = engine.projection.project_single_species_batch(
        observed_matrix=observed_matrix,
        species_values=species_values,
        fit_scale=True,
        fit_offset=True,
    )

    times = []

    for _ in range(n_repeats):
        start = time.perf_counter()

        result = engine.projection.project_single_species_batch(
            observed_matrix=observed_matrix,
            species_values=species_values,
            fit_scale=True,
            fit_offset=True,
        )

        end = time.perf_counter()
        times.append(end - start)

    return {
        "engine_name": engine_name,
        "available": True,
        "benchmark": "project_single_species_batch",
        "n_timepoints": int(species_values.shape[0]),
        "n_observables": int(observed_matrix.shape[1]),
        "n_repeats": int(n_repeats),
        "min_seconds": float(np.min(times)),
        "mean_seconds": float(np.mean(times)),
        "median_seconds": float(np.median(times)),
        "max_seconds": float(np.max(times)),
        "rss": float(result.rss),
        "warmup_rss": float(warmup_result.rss),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark production projection engine batch methods."
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
        default=1000,
    )

    parser.add_argument(
        "--n-repeats",
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
        default="benchmarks/projection_engine_batch_methods",
    )

    args = parser.parse_args()

    species_values, observed_matrix = make_projection_data(
        n_timepoints=args.n_timepoints,
        n_observables=args.n_observables,
        random_seed=args.random_seed,
    )

    rows = []

    for engine_name in args.engines:
        try:
            row = time_engine_batch_projection(
                engine_name=engine_name,
                species_values=species_values,
                observed_matrix=observed_matrix,
                n_repeats=args.n_repeats,
            )

            print(
                f"{engine_name}: "
                f"median={row['median_seconds']:.8f}s "
                f"mean={row['mean_seconds']:.8f}s "
                f"rss={row['rss']:.8g}"
            )

        except Exception as exc:
            row = {
                "engine_name": engine_name,
                "available": False,
                "benchmark": "project_single_species_batch",
                "n_timepoints": int(species_values.shape[0]),
                "n_observables": int(observed_matrix.shape[1]),
                "n_repeats": int(args.n_repeats),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }

            print(
                f"{engine_name}: unavailable "
                f"({type(exc).__name__}: {exc})"
            )

        rows.append(row)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "projection_engine_batch_method_benchmarks.json"
    csv_path = output_dir / "projection_engine_batch_method_benchmarks.csv"

    with json_path.open("w") as handle:
        json.dump(rows, handle, indent=2)

    pd.DataFrame(rows).to_csv(csv_path, index=False)

    print("\nWrote JSON:", json_path)
    print("Wrote CSV:", csv_path)


if __name__ == "__main__":
    main()
