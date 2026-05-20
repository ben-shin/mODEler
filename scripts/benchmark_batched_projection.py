from __future__ import annotations

import argparse

from odefit.benchmarking.batched_projection import (
    export_batched_projection_benchmarks,
    run_batched_projection_benchmarks,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark batched observable projection methods."
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
        "--no-loop",
        action="store_true",
        help="Skip slow Python/NumPy loop baseline.",
    )

    parser.add_argument(
        "--no-jax",
        action="store_true",
        help="Skip JAX benchmark even if JAX is installed.",
    )

    parser.add_argument(
        "--output-dir",
        default="benchmarks/batched_projection",
    )

    args = parser.parse_args()

    results = run_batched_projection_benchmarks(
        n_timepoints=args.n_timepoints,
        n_observables=args.n_observables,
        n_repeats=args.n_repeats,
        include_loop=not args.no_loop,
        include_jax=not args.no_jax,
    )

    written_files = export_batched_projection_benchmarks(
        results=results,
        output_dir=args.output_dir,
    )

    print("\nBatched projection benchmark results")
    print("====================================")

    for result in results:
        print(
            f"{result.name}: "
            f"median={result.median_seconds:.8f}s "
            f"mean={result.mean_seconds:.8f}s "
            f"n_timepoints={result.n_timepoints} "
            f"n_observables={result.n_observables} "
            f"n_repeats={result.n_repeats}"
        )

    print("\nWritten files:")
    for name, path in written_files.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
