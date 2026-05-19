from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from odefit.data.dataset import Dataset
from odefit.fitting.fit_result import FitResult
from odefit.fitting.fit_settings import FitSettings
from odefit.fitting.global_observables import (
    GlobalObservableFitOutput,
    fit_global_observable_model,
)
from odefit.fitting.initial_condition_spec import InitialConditionSpec
from odefit.fitting.model_comparison import build_ranked_model_comparison_table
from odefit.fitting.observable_spec import ObservableSpec
from odefit.fitting.parameter_spec import ParameterSpec
from odefit.model.model_spec import ModelSpec, build_model_spec


@dataclass
class GlobalObservableModelComparisonFailure:
    """
    Information about one failed model in global observable model comparison.
    """

    model_name: str
    error_type: str
    error_message: str


@dataclass
class GlobalObservableModelComparisonResult:
    """
    Result from fitting and ranking several global observable models.

    comparison_table:
        Ranked table containing one row per successful model.

    fit_outputs:
        Mapping from model name to GlobalObservableFitOutput.

    fit_results:
        Mapping from model name to FitResult.

    failures:
        Failed models, if any.

    best_model_name:
        Name of the top-ranked model.

    best_fit_result:
        FitResult for the top-ranked model.
    """

    comparison_table: pd.DataFrame
    fit_outputs: dict[str, GlobalObservableFitOutput]
    fit_results: dict[str, FitResult]
    failures: list[GlobalObservableModelComparisonFailure]
    best_model_name: str
    best_fit_result: FitResult


def build_model_specs_from_texts(
    model_texts: dict[str, str],
) -> dict[str, ModelSpec]:
    """
    Build named ModelSpec objects from a dictionary of model text.

    Example:
        {
            "irreversible": "A>B",
            "reversible": "A-B",
        }
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
    Return a model-specific value if value is a dict, otherwise return value.

    This lets callers pass either:
        observed_species_by_model={"model1": "A", "model2": "M"}

    or:
        observed_species_by_model="A"
    """

    if isinstance(value, dict):
        return value.get(model_name, default)

    if value is None:
        return default

    return value


def fit_global_observable_model_comparison(
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
    sort_by: str = "aic",
    raise_on_failure: bool = False,
) -> GlobalObservableModelComparisonResult:
    """
    Fit multiple mechanisms to the same global observable dataset and rank them.

    This is intended for HSQC-style model comparison.

    Example models:
        A>B
        A-B
        2A>A2
        2A<->A2

    Each model is fit using the same observed signal columns, but each model can
    have its own parameter specs, initial condition specs, and observed species.

    Default observable model:

        signal_i(t) = scale_i * observed_species(t) + offset_i
    """

    if not models:
        raise ValueError("At least one model is required")

    fit_outputs: dict[str, GlobalObservableFitOutput] = {}
    fit_results: dict[str, FitResult] = {}
    failures: list[GlobalObservableModelComparisonFailure] = []

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

            output = fit_global_observable_model(
                model=model,
                dataset=dataset,
                parameter_specs=parameter_specs_by_model[model_name],
                initial_condition_specs=initial_condition_specs_by_model[model_name],
                observed_species=observed_species,
                settings=settings,
                signal_columns=signal_columns,
                observable_specs=observable_specs,
                fit_scale=fit_scale,
                fit_offset=fit_offset,
                scale_initial_guess=scale_initial_guess,
                scale_lower_bound=scale_lower_bound,
                scale_upper_bound=scale_upper_bound,
                offset_initial_guess=offset_initial_guess,
                offset_lower_bound=offset_lower_bound,
                offset_upper_bound=offset_upper_bound,
            )

            fit_outputs[model_name] = output
            fit_results[model_name] = output.fit_result

        except Exception as error:
            failures.append(
                GlobalObservableModelComparisonFailure(
                    model_name=model_name,
                    error_type=type(error).__name__,
                    error_message=str(error),
                )
            )

            if raise_on_failure:
                raise

    if not fit_results:
        raise RuntimeError("All global observable model comparison fits failed")

    comparison_table = build_ranked_model_comparison_table(
        fit_results=fit_results,
        sort_by=sort_by,
    )

    best_model_name = str(comparison_table.iloc[0]["model"])

    return GlobalObservableModelComparisonResult(
        comparison_table=comparison_table,
        fit_outputs=fit_outputs,
        fit_results=fit_results,
        failures=failures,
        best_model_name=best_model_name,
        best_fit_result=fit_results[best_model_name],
    )


def export_global_observable_model_comparison(
    result: GlobalObservableModelComparisonResult,
    output_dir: str | Path,
) -> dict[str, Path]:
    """
    Export global observable model comparison summary files.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    written_files: dict[str, Path] = {}

    comparison_path = output_path / "global_observable_model_comparison.csv"

    result.comparison_table.to_csv(
        comparison_path,
        index=False,
    )

    written_files["global_observable_model_comparison"] = comparison_path

    failure_rows = [
        {
            "model": failure.model_name,
            "error_type": failure.error_type,
            "error_message": failure.error_message,
        }
        for failure in result.failures
    ]

    failures_path = output_path / "global_observable_model_comparison_failures.csv"

    pd.DataFrame(failure_rows).to_csv(
        failures_path,
        index=False,
    )

    written_files["global_observable_model_comparison_failures"] = failures_path

    return written_files
