from __future__ import annotations

import argparse
from pathlib import Path

from odefit.benchmarking.acceleration_targets import (
    export_benchmark_results,
    run_acceleration_target_benchmarks,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run mODEler acceleration target benchmarks."
    )

    parser.add_argument(
        "--output-dir",
        default="benchmarks/acceleration_targets",
        help="Output directory for benchmark results.",
    )

    parser.add_argument(
        "--include-slow",
        action="store_true",
        help="Include bootstrap and profile likelihood benchmarks.",
    )

    parser.add_argument(
        "--n-repeats-fast",
        type=int,
        default=5,
        help="Repeats for small fast benchmarks.",
    )

    parser.add_argument(
        "--n-repeats-fit",
        type=int,
        default=2,
        help="Repeats for full fit benchmarks.",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    results = run_acceleration_target_benchmarks(
        work_dir=output_dir / "work",
        n_repeats_fast=args.n_repeats_fast,
        n_repeats_fit=args.n_repeats_fit,
        include_slow=args.include_slow,
    )

    written_files = export_benchmark_results(
        results=results,
        output_dir=output_dir,
    )

    print("\nAcceleration target benchmark results")
    print("=====================================")

    for result in results:
        print(
            f"{result.name}: "
            f"median={result.median_seconds:.6f}s "
            f"mean={result.mean_seconds:.6f}s "
            f"n={result.n_repeats}"
        )

    print("\nWritten files:")
    for name, path in written_files.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
