from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SmokeCommand:
    name: str
    command: list[str]


@dataclass
class SmokeResult:
    name: str
    command: list[str]
    returncode: int
    stdout: str
    stderr: str

    @property
    def success(self) -> bool:
        return self.returncode == 0


def build_smoke_commands(
    *,
    python_executable: str,
    include_slow: bool = False,
) -> list[SmokeCommand]:
    commands = [
        SmokeCommand(
            name="single_species_variable_projection_fit",
            command=[
                python_executable,
                "-m",
                "odefit.cli",
                "fit-global-observables",
                "--config",
                "examples/configs/global_hsqc_variable_projection_config.json",
                "--variable-projection",
            ],
        ),
        SmokeCommand(
            name="single_species_variable_projection_multistart",
            command=[
                python_executable,
                "-m",
                "odefit.cli",
                "multistart-global-observables",
                "--config",
                "examples/configs/global_hsqc_variable_projection_multistart_config.json",
                "--variable-projection",
            ],
        ),
        SmokeCommand(
            name="single_species_variable_projection_model_comparison",
            command=[
                python_executable,
                "-m",
                "odefit.cli",
                "compare-global-observables",
                "--config",
                "examples/configs/global_hsqc_variable_projection_model_comparison_config.json",
                "--variable-projection",
            ],
        ),
        SmokeCommand(
            name="single_species_variable_projection_multistart_model_comparison",
            command=[
                python_executable,
                "-m",
                "odefit.cli",
                "multistart-compare-global-observables",
                "--config",
                (
                    "examples/configs/"
                    "global_hsqc_variable_projection_multistart_model_comparison_config.json"
                ),
                "--variable-projection",
            ],
        ),
        SmokeCommand(
            name="multispecies_variable_projection_fit",
            command=[
                python_executable,
                "-m",
                "odefit.cli",
                "fit-global-observables",
                "--config",
                "examples/configs/global_hsqc_multispecies_variable_projection_config.json",
            ],
        ),
        SmokeCommand(
            name="multispecies_variable_projection_multistart",
            command=[
                python_executable,
                "-m",
                "odefit.cli",
                "fit-global-observables",
                "--config",
                (
                    "examples/configs/"
                    "global_hsqc_multispecies_variable_projection_multistart_config.json"
                ),
            ],
        ),
        SmokeCommand(
            name="multispecies_variable_projection_model_comparison",
            command=[
                python_executable,
                "-m",
                "odefit.cli",
                "compare-global-observables",
                "--config",
                (
                    "examples/configs/"
                    "global_hsqc_multispecies_variable_projection_model_comparison_config.json"
                ),
            ],
        ),
        SmokeCommand(
            name="multispecies_variable_projection_multistart_model_comparison",
            command=[
                python_executable,
                "-m",
                "odefit.cli",
                "multistart-compare-global-observables",
                "--config",
                (
                    "examples/configs/"
                    "global_hsqc_multispecies_variable_projection_multistart_model_comparison_config.json"
                ),
            ],
        ),
    ]

    if include_slow:
        commands.extend(
            [
                SmokeCommand(
                    name="single_species_variable_projection_bootstrap",
                    command=[
                        python_executable,
                        "-m",
                        "odefit.cli",
                        "bootstrap-global-observables",
                        "--config",
                        "examples/configs/global_hsqc_variable_projection_bootstrap_config.json",
                        "--variable-projection",
                    ],
                ),
                SmokeCommand(
                    name="single_species_variable_projection_profile_likelihood",
                    command=[
                        python_executable,
                        "-m",
                        "odefit.cli",
                        "profile-likelihood-global-observables",
                        "--config",
                        (
                            "examples/configs/"
                            "global_hsqc_variable_projection_profile_likelihood_config.json"
                        ),
                        "--variable-projection",
                    ],
                ),
                SmokeCommand(
                    name="multispecies_variable_projection_bootstrap",
                    command=[
                        python_executable,
                        "-m",
                        "odefit.cli",
                        "bootstrap-global-observables",
                        "--config",
                        (
                            "examples/configs/"
                            "global_hsqc_multispecies_variable_projection_bootstrap_config.json"
                        ),
                    ],
                ),
                SmokeCommand(
                    name="multispecies_variable_projection_profile_likelihood",
                    command=[
                        python_executable,
                        "-m",
                        "odefit.cli",
                        "profile-likelihood-global-observables",
                        "--config",
                        (
                            "examples/configs/"
                            "global_hsqc_multispecies_variable_projection_profile_likelihood_config.json"
                        ),
                    ],
                ),
            ]
        )

    return commands


def run_command(command: SmokeCommand, *, repo_root: Path) -> SmokeResult:
    process = subprocess.run(
        command.command,
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    return SmokeResult(
        name=command.name,
        command=command.command,
        returncode=process.returncode,
        stdout=process.stdout,
        stderr=process.stderr,
    )


def run_smoke_tests(
    *,
    repo_root: Path,
    include_slow: bool = False,
    stop_on_failure: bool = False,
    python_executable: str | None = None,
) -> list[SmokeResult]:
    if python_executable is None:
        python_executable = sys.executable

    commands = build_smoke_commands(
        python_executable=python_executable,
        include_slow=include_slow,
    )

    results: list[SmokeResult] = []

    for index, command in enumerate(commands, start=1):
        print(f"\n[{index}/{len(commands)}] {command.name}")
        print(" ".join(command.command))

        result = run_command(
            command,
            repo_root=repo_root,
        )

        results.append(result)

        if result.success:
            print(f"PASS: {command.name}")
        else:
            print(f"FAIL: {command.name}")
            print("\nSTDOUT:")
            print(result.stdout)
            print("\nSTDERR:")
            print(result.stderr)

            if stop_on_failure:
                break

    return results


def print_summary(results: list[SmokeResult]) -> None:
    passed = sum(result.success for result in results)
    failed = len(results) - passed

    print("\nSmoke test summary")
    print("==================")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if failed:
        print("\nFailed workflows:")
        for result in results:
            if not result.success:
                print(f"  - {result.name}")

    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run mODEler backend workflow smoke tests."
    )

    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root. Defaults to current directory.",
    )

    parser.add_argument(
        "--include-slow",
        action="store_true",
        help="Include bootstrap and profile likelihood workflows.",
    )

    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Stop after the first failed workflow.",
    )

    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()

    results = run_smoke_tests(
        repo_root=repo_root,
        include_slow=args.include_slow,
        stop_on_failure=args.stop_on_failure,
    )

    print_summary(results)

    if any(not result.success for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
