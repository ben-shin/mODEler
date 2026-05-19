from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import traceback
from typing import Any

import pandas as pd

from odefit.data.dataset import Dataset
from odefit.fitting.fit_settings import FitSettings
from odefit.fitting.initial_condition_spec import InitialConditionSpec
from odefit.fitting.parameter_spec import ParameterSpec
from odefit.fitting.variable_projection_multistart import (
    VariableProjectionMultistartResult,
    fit_global_observable_variable_projection_multistart,
    export_variable_projection_multistart_summary,
)
from odefit.model.model_spec import ModelSpec


@dataclass
class VariableProjectionMultistartModelFailure:
    model_name: str
    error_type: str
    error_message: str
    traceback: str


@dataclass
class VariableProjectionMultistartModelComparisonResult:
    model_results: dict[str, VariableProjectionMultistartResult]
    comparison_table: pd.DataFrame
    best_model_name: str
    best_result: Any
    best_fit_result: Any
    failures: list[VariableProjectionMultistartModelFailure]

    @property
    def fit_outputs(self) -> dict[str, VariableProjectionMultistartResult]:
        return self.model_results


def _resolve_observed_species(
    observed_species_by_model: str | dict[str, str],
    model_name: str,
) -> str:
    if isinstance(observed_species_by_model, dict):
        return observed_species_by_model.get(model_name, "A")

    return observed_species_by_model


def _get_statistic(result: Any, name: str) -> float:
    if hasattr(result, "statistics") and name in result.statistics:
        return float(result.statistics[name])

    if hasattr(result, "fit_statistics") and name in result.fit_statistics:
        return float(result.fit_statistics[name])

    raise KeyError(f"Could not find statistic '{name}' on fit result.")


def fit_global_observable_variable_projection_multistart_model_comparison(
    *,
    models: dict[str, ModelSpec],
    dataset: Dataset,
    parameter_specs_by_model: dict[str, list[ParameterSpec]],
    initial_condition_specs_by_model: dict[str, list[InitialConditionSpec]],
    observed_species_by_model: str | dict[str, str],
    settings: FitSettings | dict[str, FitSettings],
    signal_columns: list[str] | None = None,
    fit_scale: bool = True,
    fit_offset: bool = True,
    backend: str = "numpy",
    method: str = "LSODA",
    n_starts: int = 10,
    random_seed: int | None = None,
    sort_by: str = "bic",
    multistart_sort_by: str = "bic",
    log_uniform: bool = True,
    show_progress: bool = True,
) -> VariableProjectionMultistartModelComparisonResult:
    """
    Compare candidate global observable mechanisms using variable projection
    multistart fitting.

    For each model:
        - run variable-projection multistart
        - keep the model's best start

    Then:
        - rank models by rss/rmse/aic/bic
    """

    if not models:
        raise ValueError("At least one model is required for model comparison.")

    sort_by = sort_by.lower()
    multistart_sort_by = multistart_sort_by.lower()

    valid_sort_fields = {"rss", "rmse", "aic", "bic"}

    if sort_by not in valid_sort_fields:
        raise ValueError(f"sort_by must be one of {sorted(valid_sort_fields)}")

    if multistart_sort_by not in valid_sort_fields:
        raise ValueError(
            f"multistart_sort_by must be one of {sorted(valid_sort_fields)}"
        )

    if signal_columns is None:
        signal_columns = dataset.signal_columns

    model_results: dict[str, VariableProjectionMultistartResult] = {}
    failures: list[VariableProjectionMultistartModelFailure] = []
    rows = []

    for model_index, (model_name, model) in enumerate(models.items(), start=1):
        try:
            model_settings = (
                settings[model_name]
                if isinstance(settings, dict)
                else settings
            )

            observed_species = _resolve_observed_species(
                observed_species_by_model=observed_species_by_model,
                model_name=model_name,
            )

            model_seed = None
            if random_seed is not None:
                model_seed = int(random_seed) + model_index - 1

            result = fit_global_observable_variable_projection_multistart(
                model=model,
                dataset=dataset,
                parameter_specs=parameter_specs_by_model[model_name],
                initial_condition_specs=initial_condition_specs_by_model[model_name],
                observed_species=observed_species,
                settings=model_settings,
                signal_columns=signal_columns,
                fit_scale=fit_scale,
                fit_offset=fit_offset,
                backend=backend,
                method=method,
                n_starts=n_starts,
                random_seed=model_seed,
                sort_by=multistart_sort_by,
                log_uniform=log_uniform,
                show_progress=show_progress,
                progress_label=(
                    f"Variable projection multistart model comparison: {model_name}"
                ),
            )

            model_results[model_name] = result

            best_fit = result.best_result

            rows.append(
                {
                    "model_name": model_name,
                    "rank": None,
                    "success": bool(best_fit.success),
                    "best_start_index": result.best_index,
                    "n_starts": n_starts,
                    "n_successful_starts": len(result.results),
                    "n_failed_starts": len(result.failures),
                    "observed_species": observed_species,
                    "rss": _get_statistic(best_fit, "rss"),
                    "rmse": _get_statistic(best_fit, "rmse"),
                    "aic": _get_statistic(best_fit, "aic"),
                    "bic": _get_statistic(best_fit, "bic"),
                    "message": best_fit.message,
                }
            )

        except Exception as exc:
            failures.append(
                VariableProjectionMultistartModelFailure(
                    model_name=model_name,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    traceback=traceback.format_exc(),
                )
            )

            rows.append(
                {
                    "model_name": model_name,
                    "rank": None,
                    "success": bool(best_fit.success),
                    "best_start_index": getattr(result, "best_index", None),
                    "n_starts": n_starts,
                    "n_successful_starts": getattr(
                        result,
                        "n_successful_starts",
                        None,
                    ),
                    "n_failed_starts": getattr(
                        result,
                        "n_failed_starts",
                        None,
                    ),
                    "observed_species": observed_species,
                    "rss": _get_statistic(best_fit, "rss"),
                    "rmse": _get_statistic(best_fit, "rmse"),
                    "aic": _get_statistic(best_fit, "aic"),
                    "bic": _get_statistic(best_fit, "bic"),
                    "message": best_fit.message,
                }
            )
        

    comparison_table = pd.DataFrame(rows)

    successful_table = comparison_table[comparison_table["success"]].copy()

    if successful_table.empty:
        print("\n=== FAILURES ===")

        for failure in failures:
            print("\nMODEL:", failure.model_name)
            print("TYPE:", failure.error_type)
            print("MESSAGE:", failure.error_message)
            print("TRACEBACK:")
            print(failure.traceback)
        
        raise RuntimeError(
            "All variable projection multistart model comparison fits failed."
        )

    comparison_table = comparison_table.sort_values(
        by=[sort_by, "model_name"],
        ascending=[True, True],
    ).reset_index(drop=True)

    comparison_table["rank"] = range(1, len(comparison_table) + 1)

    best_model_name = str(comparison_table.iloc[0]["model_name"])
    best_result = model_results[best_model_name]
    best_fit_result = best_result.best_result

    return VariableProjectionMultistartModelComparisonResult(
        model_results=model_results,
        comparison_table=comparison_table,
        best_model_name=best_model_name,
        best_result=best_result,
        best_fit_result=best_fit_result,
        failures=failures,
    )


def export_variable_projection_multistart_model_comparison(
    *,
    result: VariableProjectionMultistartModelComparisonResult,
    output_dir: str | Path,
    export_per_model_summaries: bool = True,
) -> dict[str, Path]:
    """
    Export variable-projection multistart model comparison outputs.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    written_files: dict[str, Path] = {}

    comparison_path = (
        output_path
        / "variable_projection_multistart_model_comparison.csv"
    )

    result.comparison_table.to_csv(comparison_path, index=False)
    written_files["model_comparison"] = comparison_path

    summary = {
        "best_model_name": result.best_model_name,
        "models": list(result.model_results.keys()),
        "n_models": len(result.model_results),
        "n_failed_models": len(result.failures),
    }

    summary_path = (
        output_path
        / "variable_projection_multistart_model_comparison_summary.json"
    )

    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2)

    written_files["summary"] = summary_path

    if result.failures:
        failure_rows = [
            {
                "model_name": failure.model_name,
                "error_type": failure.error_type,
                "error_message": failure.error_message,
                "traceback": failure.traceback,
            }
            for failure in result.failures
        ]

        failures_path = (
            output_path
            / "variable_projection_multistart_model_comparison_failures.csv"
        )

        pd.DataFrame(failure_rows).to_csv(failures_path, index=False)
        written_files["failures"] = failures_path

    if export_per_model_summaries:
        per_model_dir = output_path / "per_model"
        per_model_dir.mkdir(parents=True, exist_ok=True)

        for model_name, model_result in result.model_results.items():
            model_output_dir = per_model_dir / model_name

            per_model_files = export_variable_projection_multistart_summary(
                result=model_result,
                output_dir=model_output_dir,
                export_best_fit=True,
            )

            for file_name, file_path in per_model_files.items():
                written_files[f"{model_name}_{file_name}"] = Path(file_path)

    return written_files
