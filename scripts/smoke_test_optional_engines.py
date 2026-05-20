from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from odefit.engines.jax_projection import is_jax_available
from odefit.engines.numba_projection import is_numba_available


@dataclass
class OptionalEngineWorkflow:
    name: str
    command: list[str]
    required: bool = False


def build_optional_engine_workflows() -> list[OptionalEngineWorkflow]:
    workflows: list[OptionalEngineWorkflow] = []

    if is_numba_available():
        workflows.extend(
            [
                OptionalEngineWorkflow(
                    name="numba_projection_unit_tests",
                    command=[
                        sys.executable,
                        "-m",
                        "pytest",
                        "tests/test_numba_projection_engine.py",
                    ],
                ),
                OptionalEngineWorkflow(
                    name="numba_batched_projection_engine",
                    command=[
                        sys.executable,
                        "-m",
                        "pytest",
                        "tests/test_numba_batched_projection_engine.py",
                    ],
                ),
                OptionalEngineWorkflow(
                    name="numba_projection_api_workflow",
                    command=[
                        sys.executable,
                        "-m",
                        "pytest",
                        "tests/test_numba_projection_workflow.py",
                    ],
                ),
                OptionalEngineWorkflow(
                    name="projection_engine_benchmark",
                    command=[
                        sys.executable,
                        "scripts/benchmark_projection_engines.py",
                        "--n-repeats",
                        "100",
                    ],
                ),
            ]
        )

    if is_jax_available():
        workflows.extend(
            [
                OptionalEngineWorkflow(
                    name="jax_projection_unit_tests",
                    command=[
                        sys.executable,
                        "-m",
                        "pytest",
                        "tests/test_jax_projection_engine.py",
                    ],
                ),
                OptionalEngineWorkflow(
                    name="jax_projection_api_workflow",
                    command=[
                        sys.executable,
                        "-m",
                        "pytest",
                        "tests/test_jax_projection_workflow.py",
                    ],
                ),
                OptionalEngineWorkflow(
                    name="jax_projection_benchmark",
                    command=[
                        sys.executable,
                        "scripts/benchmark_projection_engines.py",
                        "--engines",
                        "reference",
                        "jax_projection",
                        "--n-repeats",
                        "100",
                    ],
                ),
                OptionalEngineWorkflow(
                    name="jax_batched_projection_engine",
                    command=[
                        sys.executable,
                        "-m",
                        "pytest",
                        "tests/test_jax_batched_projection_engine.py",
                    ],
                ),
            ]
        )

    return workflows


def run_workflow(workflow: OptionalEngineWorkflow) -> bool:
    print(f"\nRunning optional engine workflow: {workflow.name}")
    print(" ".join(workflow.command))

    result = subprocess.run(
        workflow.command,
        text=True,
        capture_output=True,
        check=False,
    )

    if result.stdout:
        print("\nSTDOUT:")
        print(result.stdout)

    if result.stderr:
        print("\nSTDERR:")
        print(result.stderr)

    if result.returncode == 0:
        print(f"PASS: {workflow.name}")
        return True

    print(f"FAIL: {workflow.name}")
    return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run optional backend engine smoke tests."
    )

    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Stop after the first failing optional engine workflow.",
    )

    args = parser.parse_args()

    workflows = build_optional_engine_workflows()

    if not workflows:
        print("No optional engine workflows are available.")
        print("Currently this usually means numba is not installed.")
        return

    passed = 0
    failed = 0
    failed_names: list[str] = []

    for workflow in workflows:
        ok = run_workflow(workflow)

        if ok:
            passed += 1
        else:
            failed += 1
            failed_names.append(workflow.name)

            if args.stop_on_failure:
                break

    print("\nOptional engine smoke test summary")
    print("==================================")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if failed_names:
        print("\nFailed workflows:")
        for name in failed_names:
            print(f"  - {name}")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
