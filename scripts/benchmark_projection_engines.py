from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from odefit.engines.registry import get_engine_bundle


def benchmark_single_species_projection(
    *,
    engine_name: str,
    n_timepoints: int,
    n_repeats: int,
) -> dict:
    engine = get_engine_bundle(engine_name)

    time = np.linspace(0.0, 10.0, n_timepoints)
    species_values = np.exp(-0.4 * time)
    observed_values = 2.0 * species_values + 0.1

    # Warmup, important for JIT engines.
    engine.projection.project_single_species(
        observed_values=observed_values,
        species_values=species_values,
        fit_scale=True,
        fit_offset=True,
    )

    times = []

    for _ in range(n_repeats):
        start = time_module()
        engine.projection.project_single_species(
            observed_values=observed_values,
            species_values=species_values,
            fit_scale=True,
            fit_offset=True,
        )
        end = time_module()
        times.append(end - start)

    return {
        "engine_name": engine_name,
        "benchmark": "single_species_projection",
        "n_timepoints": n_timepoints,
        "n_repeats": n_repeats,
        "min_seconds": min(times),
        "mean_seconds": float(np.mean(times)),
        "median_seconds": float(np.median(times)),
        "max_seconds": max(times),
    }


def time_module() -> float:
    return time.perf_counter()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark projection engines."
    )

    parser.add_argument(
        "--engines",
        nargs="+",
        default=["reference", "numba_projection", "jax_projection"],
        help="Engine names to benchmark.",
    )

    parser.add_argument(
        "--n-timepoints",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--n-repeats",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--output-dir",
        default="benchmarks/projection_engines",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for engine_name in args.engines:
        try:
            row = benchmark_single_species_projection(
                engine_name=engine_name,
                n_timepoints=args.n_timepoints,
                n_repeats=args.n_repeats,
            )
            rows.append(row)

            print(
                f"{engine_name}: "
                f"median={row['median_seconds']:.8f}s "
                f"mean={row['mean_seconds']:.8f}s"
            )

        except Exception as exc:
            row = {
                "engine_name": engine_name,
                "benchmark": "single_species_projection",
                "available": False,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
            rows.append(row)
            print(f"{engine_name}: unavailable ({type(exc).__name__}: {exc})")

    json_path = output_dir / "projection_engine_benchmarks.json"
    csv_path = output_dir / "projection_engine_benchmarks.csv"

    with json_path.open("w") as handle:
        json.dump(rows, handle, indent=2)

    pd.DataFrame(rows).to_csv(csv_path, index=False)

    print(f"\nWrote JSON: {json_path}")
    print(f"Wrote CSV: {csv_path}")


if __name__ == "__main__":
    main()
