from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
from odefit.model.model_spec import ModelSpec, build_model_spec


@dataclass
class VariableProjectionModelComparisonFailure:
    """
    Information about one failed model in variable-projection model comparison.
    """

    model_name: str
    error_type: str
    error_message: str


@dataclass
class VariableProjectionModelComparisonResult:
    """
    Result from fitting and ranking several variable-projection global observable models.
    """

    comparison_table: pd.DataFrame
    fit_results: dict[str, VariableProjectionGlobalObservableFitResult]
    failures: list[VariableProjectionModelComparisonFailure]
    best_model_name: str
    best_result: VariableProjectionGlobalObservableFitResult


def build_variable_projection_model_specs_from_texts(
    model_texts: dict[str, str],
) -> dict[str, ModelSpec]:
    """
    Build named ModelSpec objects from model text.
    """

    return {
        model_name: build_model_spec(
            text=model_text,
            name=model_name,
        )
        for model_name, model_text in model_texts.items()
    }


def _get_model_specific_value(
    value,
    model_name: str,
    default=None,
):
    """
    Return model-specific value if value is a dict, otherwise return value.
    """

    if isinstance(value, dict):
        return value.get(model_name, default)

    if value is None:
        return default

    return value


def build_variable_projection_model_comparison_table(
    fit_results: dict[str, VariableProjectionGlobalObservableFitResult],
    sort_by: str = "aic",
) -> pd.DataFrame:
    """
    Build ranked model comparison table from variable-projection fit results.
    """

    rows: list[dict] = []

    for model_name, result in fit_results.items():
        row: dict = {
            "model": model_name,
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
        raise ValueError(f"Cannot sort model comparison table by unknown column: {sort_by}")

    table = table.sort_values(
        by=sort_by,
        ascending=True,
    ).reset_index(drop=True)

    table.insert(
        0,
        "rank",
        range(1, len(table) + 1),
    )

    return table


def fit_global_observable_variable_projection_model_comparison(
    models: dict[str, ModelSpec],
    dataset: Dataset,
    parameter_specs_by_model: dict[str, list[ParameterSpec]],
    initial_condition_specs_by_model: dict[str, list[InitialConditionSpec]],
    observed_species_by_model: str | dict[str, str] = "A",
    settings_by_model: FitSettings | dict[str, FitSettings] | None = None,
    signal_columns: list[str] | None = None,
    fit_scale: bool = True,
    fit_offset: bool = True,
    backend: str = "numpy",
    method: str = "LSODA",
    sort_by: str = "aic",
    raise_on_failure: bool = False,
) -> VariableProjectionModelComparisonResult:
    """
    Fit multiple candidate mechanisms using variable projection and rank them.

    This is the fast model-comparison path for HSQC-style data where each peak is:

        peak_i(t) = scale_i * observed_species(t) + offset_i
    """

    if not models:
        raise ValueError("At least one model is required")

    if signal_columns is None:
        signal_columns = dataset.signal_columns

    fit_results: dict[str, VariableProjectionGlobalObservableFitResult] = {}
    failures: list[VariableProjectionModelComparisonFailure] = []

    for model_name, model in models.items():
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

            result = fit_global_observable_model_variable_projection(
                model=model,
                dataset=dataset,
                parameter_specs=parameter_specs_by_model[model_name],
                initial_condition_specs=initial_condition_specs_by_model[model_name],
                observed_species=observed_species,
                settings=settings,
                signal_columns=signal_columns,
                fit_scale=fit_scale,
                fit_offset=fit_offset,
                backend=backend,
                method=method,
            )

            fit_results[model_name] = result

        except Exception as error:
            failures.append(
                VariableProjectionModelComparisonFailure(
                    model_name=model_name,
                    error_type=type(error).__name__,
                    error_message=str(error),
                )
            )

            if raise_on_failure:
                raise

    if not fit_results:
        failure_details = "; ".join(
            (
                f"{failure.model_name}: "
                f"{failure.error_type}: "
                f"{failure.error_message}"
            )
            for failure in failures
        )

        raise RuntimeError(
            "All variable projection model comparison fits failed. "
            f"Failures: {failure_details}"
        )

    comparison_table = build_variable_projection_model_comparison_table(
        fit_results=fit_results,
        sort_by=sort_by,
    )

    best_model_name = str(comparison_table.iloc[0]["model"])

    return VariableProjectionModelComparisonResult(
        comparison_table=comparison_table,
        fit_results=fit_results,
        failures=failures,
        best_model_name=best_model_name,
        best_result=fit_results[best_model_name],
    )


def export_variable_projection_model_comparison(
    result: VariableProjectionModelComparisonResult,
    output_dir: str | Path,
    export_best_fit: bool = True,
) -> dict[str, Path]:
    """
    Export variable-projection model comparison summary files.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    written_files: dict[str, Path] = {}

    comparison_path = output_path / "variable_projection_model_comparison.csv"

    result.comparison_table.to_csv(
        comparison_path,
        index=False,
    )

    written_files["variable_projection_model_comparison"] = comparison_path

    failures_path = output_path / "variable_projection_model_comparison_failures.csv"

    failure_rows = [
        {
            "model": failure.model_name,
            "error_type": failure.error_type,
            "error_message": failure.error_message,
        }
        for failure in result.failures
    ]

    pd.DataFrame(failure_rows).to_csv(
        failures_path,
        index=False,
    )

    written_files["variable_projection_model_comparison_failures"] = failures_path

    if export_best_fit:
        best_fit_dir = output_path / "best_fit"

        best_fit_files = export_variable_projection_fit(
            result=result.best_result,
            output_dir=str(best_fit_dir),
        )

        for name, path in best_fit_files.items():
            written_files[f"best_fit_{name}"] = Path(path)

    return written_files
