from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from odefit.data.dataset import Dataset
from odefit.fitting.fit_result import FitResult
from odefit.fitting.fit_settings import FitSettings
from odefit.fitting.initial_condition_spec import InitialConditionSpec
from odefit.fitting.model_comparison import build_ranked_model_comparison_table
from odefit.fitting.multistart import (
    MultistartResult,
    parameter_specs_to_initial_guess_dict,
    sample_parameter_specs,
)
from odefit.fitting.observable_spec import ObservableSpec
from odefit.fitting.optimizer import fit_model
from odefit.fitting.parameter_spec import ParameterSpec
from odefit.model.model_spec import ModelSpec


@dataclass
class ParallelFitFailure:
    """
    Information about a failed parallel fit start.
    """

    start_index: int
    error_type: str
    error_message: str
    starting_parameters: dict[str, float]


@dataclass
class ParallelMultistartResult:
    """
    Result from parallel multistart fitting.

    successful_result contains the same shape as serial MultistartResult,
    but only for successful starts.

    failures stores failed starts separately.
    """

    successful_result: MultistartResult
    failures: list[ParallelFitFailure]
    n_submitted: int
    n_successful: int
    n_failed: int


def _fit_one_start_worker(
    start_index: int,
    model: ModelSpec,
    dataset: Dataset,
    parameter_specs: list[ParameterSpec],
    initial_condition_specs: list[InitialConditionSpec],
    settings: FitSettings,
    observable_specs: list[ObservableSpec] | None = None,
) -> tuple[int, FitResult, dict[str, float]]:
    """
    Worker function for one parallel fit.

    This must be top-level so it can be pickled by multiprocessing.
    """

    result = fit_model(
        model=model,
        dataset=dataset,
        parameter_specs=parameter_specs,
        initial_condition_specs=initial_condition_specs,
        observable_specs=observable_specs,
        settings=settings,
    )

    starting_parameters = parameter_specs_to_initial_guess_dict(parameter_specs)

    return start_index, result, starting_parameters


def build_multistart_parameter_spec_sets(
    parameter_specs: list[ParameterSpec],
    n_starts: int,
    random_seed: int | None = None,
    log_uniform: bool = True,
) -> list[list[ParameterSpec]]:
    """
    Build the parameter spec sets used for multistart fitting.

    Start 0 uses the original ParameterSpec objects.
    Remaining starts use randomized initial guesses.
    """

    if n_starts < 1:
        raise ValueError("n_starts must be at least 1")

    rng = np.random.default_rng(random_seed)

    parameter_spec_sets = [parameter_specs]

    for _ in range(1, n_starts):
        parameter_spec_sets.append(
            sample_parameter_specs(
                parameter_specs=parameter_specs,
                rng=rng,
                log_uniform=log_uniform,
            )
        )

    return parameter_spec_sets


def fit_multistart_parallel(
    model: ModelSpec,
    dataset: Dataset,
    parameter_specs: list[ParameterSpec],
    initial_condition_specs: list[InitialConditionSpec],
    settings: FitSettings,
    n_starts: int = 10,
    n_workers: int | None = None,
    random_seed: int | None = None,
    observable_specs: list[ObservableSpec] | None = None,
    sort_by: str = "aic",
    log_uniform: bool = True,
    raise_on_failure: bool = False,
) -> ParallelMultistartResult:
    """
    Run multistart fitting in parallel using ProcessPoolExecutor.

    Failed starts are collected and reported. If raise_on_failure=True,
    the first failed start raises immediately.
    """

    parameter_spec_sets = build_multistart_parameter_spec_sets(
        parameter_specs=parameter_specs,
        n_starts=n_starts,
        random_seed=random_seed,
        log_uniform=log_uniform,
    )

    completed_results: dict[int, FitResult] = {}
    starting_parameter_sets: dict[int, dict[str, float]] = {}
    failures: list[ParallelFitFailure] = []

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_start_index = {}

        for start_index, start_parameter_specs in enumerate(parameter_spec_sets):
            future = executor.submit(
                _fit_one_start_worker,
                start_index,
                model,
                dataset,
                start_parameter_specs,
                initial_condition_specs,
                settings,
                observable_specs,
            )

            future_to_start_index[future] = start_index

        for future in as_completed(future_to_start_index):
            start_index = future_to_start_index[future]

            try:
                (
                    returned_start_index,
                    result,
                    starting_parameters,
                ) = future.result()

                completed_results[returned_start_index] = result
                starting_parameter_sets[returned_start_index] = starting_parameters

            except Exception as error:
                starting_parameters = parameter_specs_to_initial_guess_dict(
                    parameter_spec_sets[start_index]
                )

                failure = ParallelFitFailure(
                    start_index=start_index,
                    error_type=type(error).__name__,
                    error_message=str(error),
                    starting_parameters=starting_parameters,
                )

                failures.append(failure)

                if raise_on_failure:
                    raise

    if not completed_results:
        raise RuntimeError("All parallel multistart fits failed")

    ordered_indices = sorted(completed_results)

    all_results = [completed_results[index] for index in ordered_indices]

    ordered_starting_parameter_sets = [
        starting_parameter_sets[index] for index in ordered_indices
    ]

    fit_result_map = {
        f"start_{index}": completed_results[index] for index in ordered_indices
    }

    comparison_table = build_ranked_model_comparison_table(
        fit_results=fit_result_map,
        sort_by=sort_by,
    )

    best_start_name = str(comparison_table.iloc[0]["model"])
    best_original_index = int(best_start_name.replace("start_", ""))

    best_result = completed_results[best_original_index]

    # best_index is reported in the original submitted-start index space.
    successful_result = MultistartResult(
        best_result=best_result,
        best_index=best_original_index,
        all_results=all_results,
        comparison_table=comparison_table,
        starting_parameter_sets=ordered_starting_parameter_sets,
    )

    return ParallelMultistartResult(
        successful_result=successful_result,
        failures=failures,
        n_submitted=n_starts,
        n_successful=len(completed_results),
        n_failed=len(failures),
    )


def export_parallel_multistart_summary(
    parallel_result: ParallelMultistartResult,
    output_dir: str | Path,
) -> dict[str, Path]:
    """
    Export parallel multistart comparison and failure summary.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    written_files: dict[str, Path] = {}

    comparison_path = output_path / "parallel_multistart_comparison.csv"

    parallel_result.successful_result.comparison_table.to_csv(
        comparison_path,
        index=False,
    )

    written_files["parallel_multistart_comparison"] = comparison_path

    failures_path = output_path / "parallel_multistart_failures.csv"

    failure_rows = [
        {
            "start_index": failure.start_index,
            "error_type": failure.error_type,
            "error_message": failure.error_message,
            "starting_parameters": failure.starting_parameters,
        }
        for failure in parallel_result.failures
    ]

    pd.DataFrame(failure_rows).to_csv(
        failures_path,
        index=False,
    )

    written_files["parallel_multistart_failures"] = failures_path

    return written_files
