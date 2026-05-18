from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from odefit.data.dataset import Dataset
from odefit.fitting.fit_result import FitResult
from odefit.fitting.fit_settings import FitSettings
from odefit.fitting.initial_condition_spec import InitialConditionSpec
from odefit.fitting.model_comparison import build_ranked_model_comparison_table
from odefit.fitting.observable_spec import ObservableSpec
from odefit.fitting.optimizer import fit_model
from odefit.fitting.parameter_spec import ParameterSpec
from odefit.model.model_spec import ModelSpec


@dataclass
class MultistartResult:
    """
    Result from running the same fit from multiple starting guesses.
    """

    best_result: FitResult
    best_index: int
    all_results: list[FitResult]
    comparison_table: pd.DataFrame
    starting_parameter_sets: list[dict[str, float]]


def sample_initial_guess(
    lower_bound: float,
    upper_bound: float,
    rng: np.random.Generator,
    log_uniform: bool = True,
) -> float:
    """
    Sample one initial guess inside bounds.

    If log_uniform is True and both bounds are positive, sample uniformly in log space.
    Otherwise, sample uniformly in linear space.
    """

    if not np.isfinite(lower_bound):
        raise ValueError("Lower bound must be finite for multistart sampling")

    if not np.isfinite(upper_bound):
        raise ValueError("Upper bound must be finite for multistart sampling")

    if lower_bound >= upper_bound:
        raise ValueError("Lower bound must be less than upper bound")

    if log_uniform and lower_bound > 0.0 and upper_bound > 0.0:
        log_lower = np.log(lower_bound)
        log_upper = np.log(upper_bound)

        return float(np.exp(rng.uniform(log_lower, log_upper)))

    return float(rng.uniform(lower_bound, upper_bound))


def parameter_specs_to_initial_guess_dict(
    parameter_specs: list[ParameterSpec],
) -> dict[str, float]:
    """
    Convert ParameterSpec objects into a simple initial-guess dictionary.
    """

    guesses: dict[str, float] = {}

    for parameter in parameter_specs:
        if parameter.fixed and parameter.fixed_value is not None:
            guesses[parameter.name] = float(parameter.fixed_value)
        else:
            guesses[parameter.name] = float(parameter.initial_guess)

    return guesses


def sample_parameter_specs(
    parameter_specs: list[ParameterSpec],
    rng: np.random.Generator,
    log_uniform: bool = True,
) -> list[ParameterSpec]:
    """
    Create a new list of ParameterSpec objects with randomized initial guesses.

    Fixed parameters and tied parameters are left unchanged.
    """

    sampled_specs = []

    for parameter in parameter_specs:
        if parameter.fixed or parameter.tied_to is not None:
            sampled_specs.append(parameter)
            continue

        sampled_guess = sample_initial_guess(
            lower_bound=parameter.lower_bound,
            upper_bound=parameter.upper_bound,
            rng=rng,
            log_uniform=log_uniform,
        )

        sampled_specs.append(
            replace(
                parameter,
                initial_guess=sampled_guess,
            )
        )

    return sampled_specs


def fit_multistart(
    model: ModelSpec,
    dataset: Dataset,
    parameter_specs: list[ParameterSpec],
    initial_condition_specs: list[InitialConditionSpec],
    settings: FitSettings,
    n_starts: int = 10,
    random_seed: int | None = None,
    observable_specs: list[ObservableSpec] | None = None,
    sort_by: str = "aic",
    log_uniform: bool = True,
) -> MultistartResult:
    """
    Run the same fit from multiple parameter starting guesses.

    The first start uses the provided ParameterSpec objects unchanged.
    Remaining starts use randomized initial guesses within bounds.
    """

    if n_starts < 1:
        raise ValueError("n_starts must be at least 1")

    rng = np.random.default_rng(random_seed)

    all_results: list[FitResult] = []
    starting_parameter_sets: list[dict[str, float]] = []

    for start_index in range(n_starts):
        if start_index == 0:
            start_parameter_specs = parameter_specs
        else:
            start_parameter_specs = sample_parameter_specs(
                parameter_specs=parameter_specs,
                rng=rng,
                log_uniform=log_uniform,
            )

        starting_parameter_sets.append(
            parameter_specs_to_initial_guess_dict(start_parameter_specs)
        )

        result = fit_model(
            model=model,
            dataset=dataset,
            parameter_specs=start_parameter_specs,
            initial_condition_specs=initial_condition_specs,
            observable_specs=observable_specs,
            settings=settings,
        )

        all_results.append(result)

    fit_result_map = {
        f"start_{index}": result for index, result in enumerate(all_results)
    }

    comparison_table = build_ranked_model_comparison_table(
        fit_results=fit_result_map,
        sort_by=sort_by,
    )

    best_start_name = str(comparison_table.iloc[0]["model"])
    best_index = int(best_start_name.replace("start_", ""))

    return MultistartResult(
        best_result=all_results[best_index],
        best_index=best_index,
        all_results=all_results,
        comparison_table=comparison_table,
        starting_parameter_sets=starting_parameter_sets,
    )


def export_multistart_comparison(
    multistart_result: MultistartResult,
    file_path: str | Path,
) -> Path:
    """
    Export the multistart comparison table to CSV.
    """

    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    multistart_result.comparison_table.to_csv(path, index=False)

    return path
