from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from odefit.data.dataset import Dataset
from odefit.fitting.fit_result import FitResult
from odefit.fitting.fit_settings import FitSettings
from odefit.fitting.global_observable_multistart import (
    GlobalObservableMultistartResult,
    export_global_observable_multistart_summary,
    fit_global_observable_multistart,
)
from odefit.fitting.global_observables import build_shared_species_observable_specs
from odefit.fitting.initial_condition_spec import InitialConditionSpec
from odefit.fitting.model_comparison import build_ranked_model_comparison_table
from odefit.fitting.observable_spec import ObservableSpec
from odefit.fitting.parameter_spec import ParameterSpec
from odefit.model.model_spec import ModelSpec


@dataclass
class GlobalObservableMultistartModelComparisonFailure:
    """
    Information about one failed model in multistart model comparison.
    """

    model_name: str
    error_type: str
    error_message: str


@dataclass
class GlobalObservableMultistartModelComparisonResult:
    """
    Result from comparing several global observable models using multistart.

    For each candidate model:
    - run global observable multistart
    - keep that model's best fit

    Then:
    - rank the best fit from each model by AIC/BIC/RSS/RMSE/etc.
    """

    comparison_table: pd.DataFrame
    multistart_results: dict[str, GlobalObservableMultistartResult]
    best_fit_results_by_model: dict[str, FitResult]
    failures: list[GlobalObservableMultistartModelComparisonFailure]
    best_model_name: str
    best_fit_result: FitResult


def _get_model_specific_value(
    value,
    model_name: str,
    default=None,
):
    """
    Return a model-specific value if value is a dict, otherwise return value.
    """

    if isinstance(value, dict):
        return value.get(model_name, default)

    if value is None:
        return default

    return value


def fit_global_observable_multistart_model_comparison(
    models: dict[str, ModelSpec],
    dataset: Dataset,
    parameter_specs_by_model: dict[str, list[ParameterSpec]],
    initial_condition_specs_by_model: dict[str, list[InitialConditionSpec]],
    observed_species_by_model: str | dict[str, str] = "A",
    settings_by_model: FitSettings | dict[str, FitSettings] | None = None,
    observable_specs_by_model: dict[str, list[ObservableSpec]] | None = None,
    signal_columns: list[str] | None = None,
    fit_scale: bool = True,
    fit_offset: bool = True,
    scale_initial_guess: float = 1.0,
    scale_lower_bound: float = 0.0,
    scale_upper_bound: float = float("inf"),
    offset_initial_guess: float = 0.0,
    offset_lower_bound: float = -float("inf"),
    offset_upper_bound: float = float("inf"),
    n_starts: int = 10,
    n_workers: int | None = 1,
    random_seed: int | None = None,
    sort_by: str = "aic",
    multistart_sort_by: str | None = None,
    log_uniform_parameters: bool = True,
    randomize_observable_scales: bool = True,
    randomize_observable_offsets: bool = True,
    log_uniform_observable_scales: bool = False,
    raise_on_failure: bool = False,
) -> GlobalObservableMultistartModelComparisonResult:
    """
    Compare several global observable mechanisms using multistart per model.

    This is the robust scientific workflow:

        for each candidate mechanism:
            run global observable multistart
            keep the model's best fit

        then:
            compare each model's best fit by AIC/BIC/RSS/RMSE/etc.

    This avoids ranking models based on a single unlucky optimizer start.
    """

    if not models:
        raise ValueError("At least one model is required")

    if signal_columns is None:
        signal_columns = dataset.signal_columns

    if multistart_sort_by is None:
        multistart_sort_by = sort_by

    multistart_results: dict[str, GlobalObservableMultistartResult] = {}
    best_fit_results_by_model: dict[str, FitResult] = {}
    failures: list[GlobalObservableMultistartModelComparisonFailure] = []

    for model_index, (model_name, model) in enumerate(models.items()):
        try:
            if model_name not in parameter_specs_by_model:
                raise ValueError(f"Missing parameter specs for model: {model_name}")

            if model_name not in initial_condition_specs_by_model:
                raise ValueError(
                    f"Missing initial condition specs for model: {model_name}"
                )

            observed_species = _get_model_specific_value(
                observed_species_by_model,
                model_name=model_name,
                default="A",
            )

            if observed_species not in model.species:
                raise ValueError(
                    f"Observed species {observed_species} is not in model: {model_name}"
                )

            settings = _get_model_specific_value(
                settings_by_model,
                model_name=model_name,
                default=None,
            )

            if settings is None:
                settings = FitSettings(
                    species_mapping={},
                )

            observable_specs = None

            if observable_specs_by_model is not None:
                observable_specs = observable_specs_by_model.get(model_name)

            if observable_specs is None:
                observable_specs = build_shared_species_observable_specs(
                    signal_columns=signal_columns,
                    species=observed_species,
                    fit_scale=fit_scale,
                    fit_offset=fit_offset,
                    scale_initial_guess=scale_initial_guess,
                    scale_lower_bound=scale_lower_bound,
                    scale_upper_bound=scale_upper_bound,
                    offset_initial_guess=offset_initial_guess,
                    offset_lower_bound=offset_lower_bound,
                    offset_upper_bound=offset_upper_bound,
                )

            model_seed = None

            if random_seed is not None:
                model_seed = random_seed + model_index

            multistart_result = fit_global_observable_multistart(
                model=model,
                dataset=dataset,
                parameter_specs=parameter_specs_by_model[model_name],
                initial_condition_specs=initial_condition_specs_by_model[model_name],
                observable_specs=observable_specs,
                settings=settings,
                n_starts=n_starts,
                n_workers=n_workers,
                random_seed=model_seed,
                sort_by=multistart_sort_by,
                log_uniform_parameters=log_uniform_parameters,
                randomize_observable_scales=randomize_observable_scales,
                randomize_observable_offsets=randomize_observable_offsets,
                log_uniform_observable_scales=log_uniform_observable_scales,
                raise_on_failure=raise_on_failure,
            )

            multistart_results[model_name] = multistart_result
            best_fit_results_by_model[model_name] = multistart_result.best_result

        except Exception as error:
            failures.append(
                GlobalObservableMultistartModelComparisonFailure(
                    model_name=model_name,
                    error_type=type(error).__name__,
                    error_message=str(error),
                )
            )

            if raise_on_failure:
                raise

    if not best_fit_results_by_model:
        failure_details = "; ".join(
            (f"{failure.model_name}: {failure.error_type}: {failure.error_message}")
            for failure in failures
        )

        raise RuntimeError(
            f"All multistart model comparison fits failed. Failures: {failure_details}"
        )

    comparison_table = build_ranked_model_comparison_table(
        fit_results=best_fit_results_by_model,
        sort_by=sort_by,
    )

    best_model_name = str(comparison_table.iloc[0]["model"])

    return GlobalObservableMultistartModelComparisonResult(
        comparison_table=comparison_table,
        multistart_results=multistart_results,
        best_fit_results_by_model=best_fit_results_by_model,
        failures=failures,
        best_model_name=best_model_name,
        best_fit_result=best_fit_results_by_model[best_model_name],
    )


def export_global_observable_multistart_model_comparison(
    result: GlobalObservableMultistartModelComparisonResult,
    output_dir: str | Path,
    export_per_model_summaries: bool = True,
) -> dict[str, Path]:
    """
    Export multistart global observable model comparison summary files.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    written_files: dict[str, Path] = {}

    comparison_path = output_path / "global_observable_multistart_model_comparison.csv"

    result.comparison_table.to_csv(
        comparison_path,
        index=False,
    )

    written_files["global_observable_multistart_model_comparison"] = comparison_path

    failure_rows = [
        {
            "model": failure.model_name,
            "error_type": failure.error_type,
            "error_message": failure.error_message,
        }
        for failure in result.failures
    ]

    failures_path = (
        output_path / "global_observable_multistart_model_comparison_failures.csv"
    )

    pd.DataFrame(failure_rows).to_csv(
        failures_path,
        index=False,
    )

    written_files["global_observable_multistart_model_comparison_failures"] = (
        failures_path
    )

    per_model_summary_rows: list[dict] = []

    for model_name, multistart_result in result.multistart_results.items():
        per_model_summary_rows.append(
            {
                "model": model_name,
                "best_start_index": multistart_result.best_index,
                "n_submitted": multistart_result.n_submitted,
                "n_successful": multistart_result.n_successful,
                "n_failed": multistart_result.n_failed,
                "best_success": multistart_result.best_result.success,
                "best_cost": multistart_result.best_result.cost,
                "best_aic": multistart_result.best_result.statistics.get("aic"),
                "best_bic": multistart_result.best_result.statistics.get("bic"),
                "best_rmse": multistart_result.best_result.statistics.get("rmse"),
                "best_rss": multistart_result.best_result.statistics.get("rss"),
            }
        )

    per_model_summary_path = (
        output_path / "global_observable_multistart_model_summary.csv"
    )

    pd.DataFrame(per_model_summary_rows).to_csv(
        per_model_summary_path,
        index=False,
    )

    written_files["global_observable_multistart_model_summary"] = per_model_summary_path

    if export_per_model_summaries:
        per_model_dir = output_path / "per_model_multistart"

        for model_name, multistart_result in result.multistart_results.items():
            safe_model_name = (
                model_name.replace("/", "_").replace("\\", "_").replace(" ", "_")
            )

            model_output_dir = per_model_dir / safe_model_name

            per_model_files = export_global_observable_multistart_summary(
                result=multistart_result,
                output_dir=model_output_dir,
            )

            for file_key, file_path in per_model_files.items():
                written_files[f"{model_name}_{file_key}"] = file_path

    return written_files
