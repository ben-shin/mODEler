from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import traceback

import numpy as np
import pandas as pd

from odefit.data.dataset import Dataset
from odefit.fitting.fit_settings import FitSettings
from odefit.fitting.initial_condition_spec import InitialConditionSpec
from odefit.fitting.multispecies_variable_projection_multistart import (
    MultispeciesVariableProjectionMultistartResult,
    export_multispecies_variable_projection_multistart_summary,
    fit_global_observable_model_multispecies_variable_projection_multistart,
)
from odefit.fitting.parameter_spec import ParameterSpec
from odefit.model.model_spec import ModelSpec


@dataclass
class MultispeciesVariableProjectionMultistartModelFailure:
    model_name: str
    error_type: str
    error_message: str
    traceback: str


@dataclass
class MultispeciesVariableProjectionMultistartModelComparisonResult:
    model_results: dict[str, MultispeciesVariableProjectionMultistartResult]
    comparison_table: pd.DataFrame
    best_model_name: str
    best_result: MultispeciesVariableProjectionMultistartResult
    best_fit_result: object
    failures: list[MultispeciesVariableProjectionMultistartModelFailure]


def _resolve_observed_species(
    observed_species_by_model: list[str] | dict[str, list[str]],
    model_name: str,
) -> list[str]:
    if isinstance(observed_species_by_model, dict):
        return observed_species_by_model[model_name]

    return observed_species_by_model


def fit_global_observable_multispecies_variable_projection_multistart_model_comparison(
    *,
    models: dict[str, ModelSpec],
    dataset: Dataset,
    parameter_specs_by_model: dict[str, list[ParameterSpec]],
    initial_condition_specs_by_model: dict[str, list[InitialConditionSpec]],
    observed_species_by_model: list[str] | dict[str, list[str]],
    settings: FitSettings | dict[str, FitSettings],
    signal_columns: list[str] | None = None,
    fit_offset: bool = True,
    backend: str = "numpy",
    method: str = "LSODA",
    n_starts: int = 10,
    random_seed: int | None = None,
    sort_by: str = "bic",
    multistart_sort_by: str | None = None,
    log_uniform: bool = True,
    show_progress: bool = True,
    engine_name: str = "reference",
) -> MultispeciesVariableProjectionMultistartModelComparisonResult:
    """
    Compare candidate mechanisms using multispecies variable projection
    multistart fitting.

    For each model:
        - run multispecies variable-projection multistart
        - keep the model's best start

    Then:
        - rank models by rss/rmse/aic/bic
    """

    if not models:
        raise ValueError("At least one model is required.")

    sort_by = sort_by.lower()

    if multistart_sort_by is None:
        multistart_sort_by = sort_by

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

    model_results: dict[str, MultispeciesVariableProjectionMultistartResult] = {}
    failures: list[MultispeciesVariableProjectionMultistartModelFailure] = []
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

            if show_progress:
                print(
                    "Multispecies variable projection multistart "
                    f"model comparison: {model_name}"
                )

            result = (
                fit_global_observable_model_multispecies_variable_projection_multistart(
                    model=model,
                    dataset=dataset,
                    parameter_specs=parameter_specs_by_model[model_name],
                    initial_condition_specs=initial_condition_specs_by_model[
                        model_name
                    ],
                    observed_species=observed_species,
                    settings=model_settings,
                    signal_columns=signal_columns,
                    fit_offset=fit_offset,
                    backend=backend,
                    method=method,
                    n_starts=n_starts,
                    random_seed=model_seed,
                    sort_by=multistart_sort_by,
                    log_uniform=log_uniform,
                    show_progress=show_progress,
                    engine_name=engine_name,
                )
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
                    "n_successful_starts": len(result.successful_results),
                    "n_failed_starts": len(result.failures),
                    "observed_species": ",".join(observed_species),
                    "rss": float(best_fit.statistics["rss"]),
                    "rmse": float(best_fit.statistics["rmse"]),
                    "aic": float(best_fit.statistics["aic"]),
                    "bic": float(best_fit.statistics["bic"]),
                    "n_points": int(best_fit.statistics["n_points"]),
                    "n_kinetic_parameters": int(
                        best_fit.statistics["n_kinetic_parameters"]
                    ),
                    "n_observable_columns": int(
                        best_fit.statistics["n_observable_columns"]
                    ),
                    "n_projected_species": int(
                        best_fit.statistics["n_projected_species"]
                    ),
                    "message": best_fit.message,
                }
            )

        except Exception as exc:
            failures.append(
                MultispeciesVariableProjectionMultistartModelFailure(
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
                    "success": False,
                    "best_start_index": None,
                    "n_starts": n_starts,
                    "n_successful_starts": 0,
                    "n_failed_starts": n_starts,
                    "observed_species": "",
                    "rss": np.inf,
                    "rmse": np.inf,
                    "aic": np.inf,
                    "bic": np.inf,
                    "n_points": 0,
                    "n_kinetic_parameters": 0,
                    "n_observable_columns": len(signal_columns),
                    "n_projected_species": 0,
                    "message": str(exc),
                }
            )

    comparison_table = pd.DataFrame(rows)

    successful_table = comparison_table[comparison_table["success"]].copy()

    if successful_table.empty:
        raise RuntimeError(
            "All multispecies variable projection multistart model "
            "comparison fits failed."
        )

    comparison_table = comparison_table.sort_values(
        by=[sort_by, "model_name"],
        ascending=[True, True],
    ).reset_index(drop=True)

    comparison_table["rank"] = range(1, len(comparison_table) + 1)

    best_model_name = str(comparison_table.iloc[0]["model_name"])
    best_result = model_results[best_model_name]
    best_fit_result = best_result.best_result

    return MultispeciesVariableProjectionMultistartModelComparisonResult(
        model_results=model_results,
        comparison_table=comparison_table,
        best_model_name=best_model_name,
        best_result=best_result,
        best_fit_result=best_fit_result,
        failures=failures,
    )


def export_multispecies_variable_projection_multistart_model_comparison(
    *,
    result: MultispeciesVariableProjectionMultistartModelComparisonResult,
    output_dir: str | Path,
    export_best_fit: bool = True,
    export_per_model_summaries: bool = True,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    written_files: dict[str, Path] = {}

    comparison_path = (
        output_path
        / "multispecies_variable_projection_multistart_model_comparison.csv"
    )
    result.comparison_table.to_csv(comparison_path, index=False)
    written_files["model_comparison"] = comparison_path

    summary_path = (
        output_path
        / "multispecies_variable_projection_multistart_model_comparison_summary.json"
    )

    summary = {
        "best_model_name": result.best_model_name,
        "n_models": len(result.comparison_table),
        "n_successful_models": len(result.model_results),
        "n_failed_models": len(result.failures),
        "best_start_index": result.best_result.best_index,
        "best_statistics": result.best_fit_result.statistics,
        "best_fitted_parameters": result.best_fit_result.fitted_parameters,
    }

    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2)

    written_files["summary"] = summary_path

    if result.failures:
        failures_path = (
            output_path
            / "multispecies_variable_projection_multistart_model_failures.csv"
        )

        pd.DataFrame(
            [
                {
                    "model_name": failure.model_name,
                    "error_type": failure.error_type,
                    "error_message": failure.error_message,
                    "traceback": failure.traceback,
                }
                for failure in result.failures
            ]
        ).to_csv(failures_path, index=False)

        written_files["failures"] = failures_path

    if export_best_fit:
        best_fit_files = export_multispecies_variable_projection_multistart_summary(
            result=result.best_result,
            output_dir=output_path / "best_model",
            export_best_fit=True,
        )

        for name, path in best_fit_files.items():
            written_files[f"best_model_{name}"] = path

    if export_per_model_summaries:
        per_model_dir = output_path / "per_model"
        per_model_dir.mkdir(parents=True, exist_ok=True)

        for model_name, model_result in result.model_results.items():
            model_files = export_multispecies_variable_projection_multistart_summary(
                result=model_result,
                output_dir=per_model_dir / model_name,
                export_best_fit=True,
            )

            for name, path in model_files.items():
                written_files[f"{model_name}_{name}"] = path

    return written_files
