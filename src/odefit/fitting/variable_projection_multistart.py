from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from odefit.data.dataset import Dataset
from odefit.fitting.fit_settings import FitSettings
from odefit.fitting.initial_condition_spec import InitialConditionSpec
from odefit.fitting.parameter_spec import ParameterSpec
from odefit.fitting.variable_projection import (
    VariableProjectionGlobalObservableFitResult,
    export_variable_projection_fit,
    fit_global_observable_model_variable_projection,
)
from odefit.model.model_spec import ModelSpec
from odefit.utils.progress import ProgressReporter


@dataclass
class VariableProjectionMultistartFailure:
    """
    Information about one failed variable-projection multistart fit.
    """

    start_index: int
    error_type: str
    error_message: str
    starting_parameters: dict[str, float]


@dataclass
class VariableProjectionMultistartResult:
    """
    Result from variable-projection global observable multistart fitting.
    """

    best_result: VariableProjectionGlobalObservableFitResult
    best_index: int
    all_results: list[VariableProjectionGlobalObservableFitResult]
    comparison_table: pd.DataFrame
    starting_parameter_sets: list[dict[str, float]]
    failures: list[VariableProjectionMultistartFailure]
    n_submitted: int
    n_successful: int
    n_failed: int


def parameter_specs_to_initial_guess_dict(
    parameter_specs: list[ParameterSpec],
) -> dict[str, float]:
    """
    Convert ParameterSpec objects into a starting-value dictionary.
    """

    guesses: dict[str, float] = {}

    for spec in parameter_specs:
        if spec.fixed:
            value = spec.fixed_value if spec.fixed_value is not None else spec.initial_guess
        else:
            value = spec.initial_guess

        guesses[spec.name] = float(value)

    return guesses


def sample_parameter_initial_guess(
    initial_guess: float,
    lower_bound: float,
    upper_bound: float,
    rng: np.random.Generator,
    log_uniform: bool = True,
) -> float:
    """
    Sample a parameter initial guess within finite bounds.
    """

    if not np.isfinite(lower_bound) or not np.isfinite(upper_bound):
        return float(initial_guess)

    if lower_bound >= upper_bound:
        raise ValueError("Parameter lower bound must be less than upper bound")

    if log_uniform and lower_bound > 0.0 and upper_bound > 0.0:
        return float(
            np.exp(
                rng.uniform(
                    np.log(lower_bound),
                    np.log(upper_bound),
                )
            )
        )

    return float(rng.uniform(lower_bound, upper_bound))


def sample_parameter_specs(
    parameter_specs: list[ParameterSpec],
    rng: np.random.Generator,
    log_uniform: bool = True,
) -> list[ParameterSpec]:
    """
    Create randomized ParameterSpec objects for one multistart run.

    Fixed and tied parameters are left unchanged.
    """

    sampled_specs: list[ParameterSpec] = []

    for spec in parameter_specs:
        if spec.fixed or spec.tied_to is not None:
            sampled_specs.append(spec)
            continue

        sampled_initial_guess = sample_parameter_initial_guess(
            initial_guess=spec.initial_guess,
            lower_bound=spec.lower_bound,
            upper_bound=spec.upper_bound,
            rng=rng,
            log_uniform=log_uniform,
        )

        sampled_specs.append(
            replace(
                spec,
                initial_guess=sampled_initial_guess,
            )
        )

    return sampled_specs


def build_variable_projection_multistart_parameter_sets(
    parameter_specs: list[ParameterSpec],
    n_starts: int,
    random_seed: int | None = None,
    log_uniform: bool = True,
) -> list[list[ParameterSpec]]:
    """
    Build parameter spec sets for variable-projection multistart.

    Start 0 preserves the original initial guesses.
    Starts 1..N use randomized free kinetic parameter guesses.
    """

    if n_starts < 1:
        raise ValueError("n_starts must be at least 1")

    rng = np.random.default_rng(random_seed)

    parameter_spec_sets: list[list[ParameterSpec]] = [parameter_specs]

    for _ in range(1, n_starts):
        parameter_spec_sets.append(
            sample_parameter_specs(
                parameter_specs=parameter_specs,
                rng=rng,
                log_uniform=log_uniform,
            )
        )

    return parameter_spec_sets


def build_variable_projection_multistart_comparison_table(
    results_by_start: dict[int, VariableProjectionGlobalObservableFitResult],
    sort_by: str = "aic",
) -> pd.DataFrame:
    """
    Build ranked comparison table for variable-projection multistart results.
    """

    rows: list[dict] = []

    for start_index, result in results_by_start.items():
        row: dict = {
            "start_index": start_index,
            "model": f"start_{start_index}",
            "success": result.success,
            "message": result.message,
            "cost": result.cost,
            "nfev": result.nfev,
            "status": result.status,
            "optimality": result.optimality,
        }

        row.update(result.statistics)

        for parameter_name, value in result.fitted_parameters.items():
            row[f"parameter_{parameter_name}"] = value

        rows.append(row)

    table = pd.DataFrame(rows)

    if table.empty:
        return table

    if sort_by not in table.columns:
        raise ValueError(f"Cannot sort multistart table by unknown column: {sort_by}")

    table = table.sort_values(
        by=sort_by,
        ascending=True,
    ).reset_index(drop=True)

    table.insert(
        0,
        "rank",
        np.arange(1, len(table) + 1),
    )

    return table


def fit_global_observable_variable_projection_multistart(
    model: ModelSpec,
    dataset: Dataset,
    parameter_specs: list[ParameterSpec],
    initial_condition_specs: list[InitialConditionSpec],
    observed_species: str = "A",
    settings: FitSettings | None = None,
    signal_columns: list[str] | None = None,
    fit_scale: bool = True,
    fit_offset: bool = True,
    backend: str = "numpy",
    method: str = "LSODA",
    n_starts: int = 10,
    random_seed: int | None = None,
    sort_by: str = "aic",
    log_uniform: bool = True,
    raise_on_failure: bool = False,
    show_progress: bool = False,
    progress_label: str = "Variable projection multistart",
) -> VariableProjectionMultistartResult:
    """
    Run variable-projection global observable multistart fitting.

    Each start randomizes only the nonlinear kinetic parameter initial guesses.
    Per-column scale/offset terms are solved analytically during each residual
    evaluation by the variable projection backend.
    """

    parameter_spec_sets = build_variable_projection_multistart_parameter_sets(
        parameter_specs=parameter_specs,
        n_starts=n_starts,
        random_seed=random_seed,
        log_uniform=log_uniform,
    )

    completed_results: dict[int, VariableProjectionGlobalObservableFitResult] = {}
    starting_parameter_sets: dict[int, dict[str, float]] = {}
    failures: list[VariableProjectionMultistartFailure] = []

    progress = ProgressReporter(
        total=n_starts,
        label=progress_label,
        enabled=show_progress,
    )

    for start_index, start_parameter_specs in enumerate(parameter_spec_sets):
        try:
            result = fit_global_observable_model_variable_projection(
                model=model,
                dataset=dataset,
                parameter_specs=start_parameter_specs,
                initial_condition_specs=initial_condition_specs,
                observed_species=observed_species,
                settings=settings,
                signal_columns=signal_columns,
                fit_scale=fit_scale,
                fit_offset=fit_offset,
                backend=backend,
                method=method,
            )

            completed_results[start_index] = result
            starting_parameter_sets[start_index] = parameter_specs_to_initial_guess_dict(
                start_parameter_specs
            )

        except Exception as error:
            failure = VariableProjectionMultistartFailure(
                start_index=start_index,
                error_type=type(error).__name__,
                error_message=str(error),
                starting_parameters=parameter_specs_to_initial_guess_dict(
                    start_parameter_specs
                ),
            )

            failures.append(failure)

            if raise_on_failure:
                raise

        finally:
            progress.update()

    progress.close()

    if not completed_results:
        failure_details = "; ".join(
            (
                f"start_{failure.start_index}: "
                f"{failure.error_type}: "
                f"{failure.error_message}"
            )
            for failure in failures
        )

        raise RuntimeError(
            "All variable projection multistart fits failed. "
            f"Failures: {failure_details}"
        )

    comparison_table = build_variable_projection_multistart_comparison_table(
        results_by_start=completed_results,
        sort_by=sort_by,
    )

    best_index = int(comparison_table.iloc[0]["start_index"])

    ordered_indices = sorted(completed_results)

    all_results = [
        completed_results[index]
        for index in ordered_indices
    ]

    ordered_starting_parameter_sets = [
        starting_parameter_sets[index]
        for index in ordered_indices
    ]

    return VariableProjectionMultistartResult(
        best_result=completed_results[best_index],
        best_index=best_index,
        all_results=all_results,
        comparison_table=comparison_table,
        starting_parameter_sets=ordered_starting_parameter_sets,
        failures=failures,
        n_submitted=n_starts,
        n_successful=len(completed_results),
        n_failed=len(failures),
    )


def export_variable_projection_multistart_summary(
    result: VariableProjectionMultistartResult,
    output_dir: str | Path,
    export_best_fit: bool = True,
) -> dict[str, Path]:
    """
    Export variable-projection multistart summary files.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    written_files: dict[str, Path] = {}

    comparison_path = output_path / "variable_projection_multistart_comparison.csv"

    result.comparison_table.to_csv(
        comparison_path,
        index=False,
    )

    written_files["variable_projection_multistart_comparison"] = comparison_path

    starting_parameters_table = pd.DataFrame(result.starting_parameter_sets)
    starting_parameters_table.insert(
        0,
        "start_index",
        list(range(len(starting_parameters_table))),
    )

    starting_parameters_path = (
        output_path / "variable_projection_multistart_starting_parameters.csv"
    )

    starting_parameters_table.to_csv(
        starting_parameters_path,
        index=False,
    )

    written_files["variable_projection_multistart_starting_parameters"] = (
        starting_parameters_path
    )

    failures_path = output_path / "variable_projection_multistart_failures.csv"

    failure_rows = [
        {
            "start_index": failure.start_index,
            "error_type": failure.error_type,
            "error_message": failure.error_message,
            "starting_parameters": failure.starting_parameters,
        }
        for failure in result.failures
    ]

    pd.DataFrame(failure_rows).to_csv(
        failures_path,
        index=False,
    )

    written_files["variable_projection_multistart_failures"] = failures_path

    if export_best_fit:
        best_fit_dir = output_path / "best_fit"

        best_fit_files = export_variable_projection_fit(
            result=result.best_result,
            output_dir=str(best_fit_dir),
        )

        for name, path in best_fit_files.items():
            written_files[f"best_fit_{name}"] = Path(path)

    return written_files
