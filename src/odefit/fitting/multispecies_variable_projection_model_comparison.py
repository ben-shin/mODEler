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
from odefit.fitting.multispecies_variable_projection import (
    MultispeciesVariableProjectionResult,
    export_multispecies_variable_projection_fit,
    fit_global_observable_model_multispecies_variable_projection,
)
from odefit.fitting.parameter_spec import ParameterSpec
from odefit.model.model_spec import ModelSpec


@dataclass
class MultispeciesVariableProjectionModelFailure:
    model_name: str
    error_type: str
    error_message: str
    traceback: str


@dataclass
class MultispeciesVariableProjectionModelComparisonResult:
    model_results: dict[str, MultispeciesVariableProjectionResult]
    comparison_table: pd.DataFrame
    best_model_name: str
    best_result: MultispeciesVariableProjectionResult
    failures: list[MultispeciesVariableProjectionModelFailure]


def _resolve_observed_species(
    observed_species_by_model: list[str] | dict[str, list[str]],
    model_name: str,
) -> list[str]:
    if isinstance(observed_species_by_model, dict):
        return observed_species_by_model[model_name]

    return observed_species_by_model


def fit_global_observable_multispecies_variable_projection_model_comparison(
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
    sort_by: str = "bic",
) -> MultispeciesVariableProjectionModelComparisonResult:
    if not models:
        raise ValueError("At least one model is required.")

    sort_by = sort_by.lower()

    if sort_by not in {"rss", "rmse", "aic", "bic"}:
        raise ValueError("sort_by must be one of: rss, rmse, aic, bic")

    if signal_columns is None:
        signal_columns = dataset.signal_columns

    model_results: dict[str, MultispeciesVariableProjectionResult] = {}
    failures: list[MultispeciesVariableProjectionModelFailure] = []
    rows = []

    for model_name, model in models.items():
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

            result = fit_global_observable_model_multispecies_variable_projection(
                model=model,
                dataset=dataset,
                parameter_specs=parameter_specs_by_model[model_name],
                initial_condition_specs=initial_condition_specs_by_model[model_name],
                observed_species=observed_species,
                settings=model_settings,
                signal_columns=signal_columns,
                fit_offset=fit_offset,
                backend=backend,
                method=method,
            )

            model_results[model_name] = result

            rows.append(
                {
                    "model_name": model_name,
                    "rank": None,
                    "success": bool(result.success),
                    "observed_species": ",".join(observed_species),
                    "rss": float(result.statistics["rss"]),
                    "rmse": float(result.statistics["rmse"]),
                    "aic": float(result.statistics["aic"]),
                    "bic": float(result.statistics["bic"]),
                    "n_points": int(result.statistics["n_points"]),
                    "n_kinetic_parameters": int(
                        result.statistics["n_kinetic_parameters"]
                    ),
                    "n_observable_columns": int(
                        result.statistics["n_observable_columns"]
                    ),
                    "n_projected_species": int(
                        result.statistics["n_projected_species"]
                    ),
                    "message": result.message,
                }
            )

        except Exception as exc:
            failures.append(
                MultispeciesVariableProjectionModelFailure(
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
            "All multispecies variable projection model comparison fits failed."
        )

    comparison_table = comparison_table.sort_values(
        by=[sort_by, "model_name"],
        ascending=[True, True],
    ).reset_index(drop=True)

    comparison_table["rank"] = range(1, len(comparison_table) + 1)

    best_model_name = str(comparison_table.iloc[0]["model_name"])

    return MultispeciesVariableProjectionModelComparisonResult(
        model_results=model_results,
        comparison_table=comparison_table,
        best_model_name=best_model_name,
        best_result=model_results[best_model_name],
        failures=failures,
    )


def export_multispecies_variable_projection_model_comparison(
    *,
    result: MultispeciesVariableProjectionModelComparisonResult,
    output_dir: str | Path,
    export_best_fit: bool = True,
    export_per_model_fits: bool = False,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    written_files: dict[str, Path] = {}

    comparison_path = (
        output_path / "multispecies_variable_projection_model_comparison.csv"
    )
    result.comparison_table.to_csv(comparison_path, index=False)
    written_files["model_comparison"] = comparison_path

    summary_path = (
        output_path / "multispecies_variable_projection_model_comparison_summary.json"
    )

    summary = {
        "best_model_name": result.best_model_name,
        "n_models": len(result.comparison_table),
        "n_successful_models": len(result.model_results),
        "n_failed_models": len(result.failures),
        "best_statistics": result.best_result.statistics,
        "best_fitted_parameters": result.best_result.fitted_parameters,
    }

    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2)

    written_files["summary"] = summary_path

    if result.failures:
        failures_path = (
            output_path / "multispecies_variable_projection_model_failures.csv"
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
        best_fit_files = export_multispecies_variable_projection_fit(
            result=result.best_result,
            output_dir=output_path / "best_fit",
        )

        for name, path in best_fit_files.items():
            written_files[f"best_fit_{name}"] = path

    if export_per_model_fits:
        per_model_dir = output_path / "per_model"
        per_model_dir.mkdir(parents=True, exist_ok=True)

        for model_name, model_result in result.model_results.items():
            model_files = export_multispecies_variable_projection_fit(
                result=model_result,
                output_dir=per_model_dir / model_name,
            )

            for name, path in model_files.items():
                written_files[f"{model_name}_{name}"] = path

    return written_files
