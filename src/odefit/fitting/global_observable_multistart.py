from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from odefit.data.dataset import Dataset
from odefit.fitting.fit_result import FitResult
from odefit.fitting.fit_settings import FitSettings
from odefit.fitting.initial_condition_spec import InitialConditionSpec
from odefit.fitting.model_comparison import build_ranked_model_comparison_table
from odefit.fitting.multistart import (
    parameter_specs_to_initial_guess_dict,
    sample_parameter_specs,
)
from odefit.fitting.observable_spec import ObservableSpec
from odefit.fitting.optimizer import fit_model
from odefit.fitting.parameter_spec import ParameterSpec
from odefit.model.model_spec import ModelSpec


@dataclass
class GlobalObservableMultistartFailure:
    """
    Information about a failed global observable multistart fit.
    """

    start_index: int
    error_type: str
    error_message: str
    starting_parameters: dict[str, float]
    starting_observables: dict[str, float]


@dataclass
class GlobalObservableMultistartResult:
    """
    Result from global observable multistart fitting.
    """

    best_result: FitResult
    best_index: int
    all_results: list[FitResult]
    comparison_table: pd.DataFrame
    starting_parameter_sets: list[dict[str, float]]
    starting_observable_sets: list[dict[str, float]]
    failures: list[GlobalObservableMultistartFailure]
    n_submitted: int
    n_successful: int
    n_failed: int


def sample_observable_initial_guess(
    initial_guess: float,
    lower_bound: float,
    upper_bound: float,
    rng: np.random.Generator,
    log_uniform: bool = False,
) -> float:
    """
    Sample an observable initial guess inside finite bounds.

    If bounds are infinite, return the original initial guess because random
    sampling over an infinite interval is not well-defined.
    """

    if not np.isfinite(lower_bound) or not np.isfinite(upper_bound):
        return float(initial_guess)

    if lower_bound >= upper_bound:
        raise ValueError("Observable lower bound must be less than upper bound")

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


def observable_specs_to_initial_guess_dict(
    observable_specs: list[ObservableSpec],
) -> dict[str, float]:
    """
    Convert ObservableSpec objects into a flat initial-guess dictionary.
    """

    guesses: dict[str, float] = {}

    for observable in observable_specs:
        if observable.scale_fixed:
            scale_value = (
                observable.scale_fixed_value
                if observable.scale_fixed_value is not None
                else observable.scale_initial_guess
            )
        else:
            scale_value = observable.scale_initial_guess

        if observable.offset_fixed:
            offset_value = (
                observable.offset_fixed_value
                if observable.offset_fixed_value is not None
                else observable.offset_initial_guess
            )
        else:
            offset_value = observable.offset_initial_guess

        guesses[f"{observable.data_column}_scale"] = float(scale_value)
        guesses[f"{observable.data_column}_offset"] = float(offset_value)

    return guesses


def sample_observable_specs(
    observable_specs: list[ObservableSpec],
    rng: np.random.Generator,
    randomize_scales: bool = True,
    randomize_offsets: bool = True,
    log_uniform_scales: bool = False,
) -> list[ObservableSpec]:
    """
    Create a new list of ObservableSpec objects with randomized scale/offset
    initial guesses.

    Fixed scale/offset values are left unchanged.
    """

    sampled_specs: list[ObservableSpec] = []

    for observable in observable_specs:
        scale_initial_guess = observable.scale_initial_guess
        offset_initial_guess = observable.offset_initial_guess

        if randomize_scales and not observable.scale_fixed:
            scale_initial_guess = sample_observable_initial_guess(
                initial_guess=observable.scale_initial_guess,
                lower_bound=observable.scale_lower_bound,
                upper_bound=observable.scale_upper_bound,
                rng=rng,
                log_uniform=log_uniform_scales,
            )

        if randomize_offsets and not observable.offset_fixed:
            offset_initial_guess = sample_observable_initial_guess(
                initial_guess=observable.offset_initial_guess,
                lower_bound=observable.offset_lower_bound,
                upper_bound=observable.offset_upper_bound,
                rng=rng,
                log_uniform=False,
            )

        sampled_specs.append(
            replace(
                observable,
                scale_initial_guess=scale_initial_guess,
                offset_initial_guess=offset_initial_guess,
            )
        )

    return sampled_specs


def build_global_observable_multistart_spec_sets(
    parameter_specs: list[ParameterSpec],
    observable_specs: list[ObservableSpec],
    n_starts: int,
    random_seed: int | None = None,
    log_uniform_parameters: bool = True,
    randomize_observable_scales: bool = True,
    randomize_observable_offsets: bool = True,
    log_uniform_observable_scales: bool = False,
) -> tuple[list[list[ParameterSpec]], list[list[ObservableSpec]]]:
    """
    Build parameter and observable spec sets for global observable multistart.

    Start 0 preserves original guesses.
    Starts 1..N use randomized guesses.
    """

    if n_starts < 1:
        raise ValueError("n_starts must be at least 1")

    rng = np.random.default_rng(random_seed)

    parameter_spec_sets: list[list[ParameterSpec]] = [parameter_specs]
    observable_spec_sets: list[list[ObservableSpec]] = [observable_specs]

    for _ in range(1, n_starts):
        parameter_spec_sets.append(
            sample_parameter_specs(
                parameter_specs=parameter_specs,
                rng=rng,
                log_uniform=log_uniform_parameters,
            )
        )

        observable_spec_sets.append(
            sample_observable_specs(
                observable_specs=observable_specs,
                rng=rng,
                randomize_scales=randomize_observable_scales,
                randomize_offsets=randomize_observable_offsets,
                log_uniform_scales=log_uniform_observable_scales,
            )
        )

    return parameter_spec_sets, observable_spec_sets


def _fit_one_global_observable_start_worker(
    start_index: int,
    model: ModelSpec,
    dataset: Dataset,
    parameter_specs: list[ParameterSpec],
    initial_condition_specs: list[InitialConditionSpec],
    observable_specs: list[ObservableSpec],
    settings: FitSettings,
) -> tuple[int, FitResult, dict[str, float], dict[str, float]]:
    """
    Worker for one global observable multistart fit.
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
    starting_observables = observable_specs_to_initial_guess_dict(observable_specs)

    return start_index, result, starting_parameters, starting_observables


def fit_global_observable_multistart(
    model: ModelSpec,
    dataset: Dataset,
    parameter_specs: list[ParameterSpec],
    initial_condition_specs: list[InitialConditionSpec],
    observable_specs: list[ObservableSpec],
    settings: FitSettings,
    n_starts: int = 10,
    n_workers: int | None = 1,
    random_seed: int | None = None,
    sort_by: str = "aic",
    log_uniform_parameters: bool = True,
    randomize_observable_scales: bool = True,
    randomize_observable_offsets: bool = True,
    log_uniform_observable_scales: bool = False,
    raise_on_failure: bool = False,
) -> GlobalObservableMultistartResult:
    """
    Run global observable multistart fitting.

    Randomizes:
    - free kinetic parameter initial guesses
    - free observable scale initial guesses
    - free observable offset initial guesses
    """

    parameter_spec_sets, observable_spec_sets = (
        build_global_observable_multistart_spec_sets(
            parameter_specs=parameter_specs,
            observable_specs=observable_specs,
            n_starts=n_starts,
            random_seed=random_seed,
            log_uniform_parameters=log_uniform_parameters,
            randomize_observable_scales=randomize_observable_scales,
            randomize_observable_offsets=randomize_observable_offsets,
            log_uniform_observable_scales=log_uniform_observable_scales,
        )
    )

    completed_results: dict[int, FitResult] = {}
    starting_parameter_sets: dict[int, dict[str, float]] = {}
    starting_observable_sets: dict[int, dict[str, float]] = {}
    failures: list[GlobalObservableMultistartFailure] = []

    if n_workers is None or n_workers <= 1:
        for start_index in range(n_starts):
            start_parameter_specs = parameter_spec_sets[start_index]
            start_observable_specs = observable_spec_sets[start_index]

            try:
                result = fit_model(
                    model=model,
                    dataset=dataset,
                    parameter_specs=start_parameter_specs,
                    initial_condition_specs=initial_condition_specs,
                    observable_specs=start_observable_specs,
                    settings=settings,
                )

                completed_results[start_index] = result
                starting_parameter_sets[start_index] = (
                    parameter_specs_to_initial_guess_dict(start_parameter_specs)
                )
                starting_observable_sets[start_index] = (
                    observable_specs_to_initial_guess_dict(start_observable_specs)
                )

            except Exception as error:
                failure = GlobalObservableMultistartFailure(
                    start_index=start_index,
                    error_type=type(error).__name__,
                    error_message=str(error),
                    starting_parameters=parameter_specs_to_initial_guess_dict(
                        start_parameter_specs
                    ),
                    starting_observables=observable_specs_to_initial_guess_dict(
                        start_observable_specs
                    ),
                )

                failures.append(failure)

                if raise_on_failure:
                    raise

    else:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            future_to_start_index = {}

            for start_index in range(n_starts):
                future = executor.submit(
                    _fit_one_global_observable_start_worker,
                    start_index,
                    model,
                    dataset,
                    parameter_spec_sets[start_index],
                    initial_condition_specs,
                    observable_spec_sets[start_index],
                    settings,
                )

                future_to_start_index[future] = start_index

            for future in as_completed(future_to_start_index):
                start_index = future_to_start_index[future]

                try:
                    (
                        returned_start_index,
                        result,
                        starting_parameters,
                        starting_observables,
                    ) = future.result()

                    completed_results[returned_start_index] = result
                    starting_parameter_sets[returned_start_index] = (
                        starting_parameters
                    )
                    starting_observable_sets[returned_start_index] = (
                        starting_observables
                    )

                except Exception as error:
                    failure = GlobalObservableMultistartFailure(
                        start_index=start_index,
                        error_type=type(error).__name__,
                        error_message=str(error),
                        starting_parameters=parameter_specs_to_initial_guess_dict(
                            parameter_spec_sets[start_index]
                        ),
                        starting_observables=observable_specs_to_initial_guess_dict(
                            observable_spec_sets[start_index]
                        ),
                    )

                    failures.append(failure)

                    if raise_on_failure:
                        raise

    if not completed_results:
        raise RuntimeError("All global observable multistart fits failed")

    ordered_indices = sorted(completed_results)

    all_results = [
        completed_results[index]
        for index in ordered_indices
    ]

    ordered_starting_parameter_sets = [
        starting_parameter_sets[index]
        for index in ordered_indices
    ]

    ordered_starting_observable_sets = [
        starting_observable_sets[index]
        for index in ordered_indices
    ]

    fit_result_map = {
        f"start_{index}": completed_results[index]
        for index in ordered_indices
    }

    comparison_table = build_ranked_model_comparison_table(
        fit_results=fit_result_map,
        sort_by=sort_by,
    )

    best_start_name = str(comparison_table.iloc[0]["model"])
    best_index = int(best_start_name.replace("start_", ""))

    return GlobalObservableMultistartResult(
        best_result=completed_results[best_index],
        best_index=best_index,
        all_results=all_results,
        comparison_table=comparison_table,
        starting_parameter_sets=ordered_starting_parameter_sets,
        starting_observable_sets=ordered_starting_observable_sets,
        failures=failures,
        n_submitted=n_starts,
        n_successful=len(completed_results),
        n_failed=len(failures),
    )


def export_global_observable_multistart_summary(
    result: GlobalObservableMultistartResult,
    output_dir: str | Path,
) -> dict[str, Path]:
    """
    Export global observable multistart summary CSV files.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    written_files: dict[str, Path] = {}

    comparison_path = output_path / "global_observable_multistart_comparison.csv"

    result.comparison_table.to_csv(
        comparison_path,
        index=False,
    )

    written_files["global_observable_multistart_comparison"] = comparison_path

    starting_parameters_table = pd.DataFrame(result.starting_parameter_sets)
    starting_parameters_table.insert(
        0,
        "start_index",
        list(range(len(starting_parameters_table))),
    )

    starting_parameters_path = (
        output_path / "global_observable_multistart_starting_parameters.csv"
    )

    starting_parameters_table.to_csv(
        starting_parameters_path,
        index=False,
    )

    written_files["global_observable_multistart_starting_parameters"] = (
        starting_parameters_path
    )

    starting_observables_table = pd.DataFrame(result.starting_observable_sets)
    starting_observables_table.insert(
        0,
        "start_index",
        list(range(len(starting_observables_table))),
    )

    starting_observables_path = (
        output_path / "global_observable_multistart_starting_observables.csv"
    )

    starting_observables_table.to_csv(
        starting_observables_path,
        index=False,
    )

    written_files["global_observable_multistart_starting_observables"] = (
        starting_observables_path
    )

    failures_path = output_path / "global_observable_multistart_failures.csv"

    failure_rows = [
        {
            "start_index": failure.start_index,
            "error_type": failure.error_type,
            "error_message": failure.error_message,
            "starting_parameters": failure.starting_parameters,
            "starting_observables": failure.starting_observables,
        }
        for failure in result.failures
    ]

    pd.DataFrame(failure_rows).to_csv(
        failures_path,
        index=False,
    )

    written_files["global_observable_multistart_failures"] = failures_path

    return written_files
