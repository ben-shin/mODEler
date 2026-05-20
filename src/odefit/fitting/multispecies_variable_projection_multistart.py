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
from odefit.fitting.parameter_spec import ParameterSpec
from odefit.fitting.multispecies_variable_projection import (
    MultispeciesVariableProjectionResult,
    export_multispecies_variable_projection_fit,
    fit_global_observable_model_multispecies_variable_projection,
)
from odefit.model.model_spec import ModelSpec


@dataclass
class MultispeciesVariableProjectionStartFailure:
    start_index: int
    error_type: str
    error_message: str
    traceback: str


@dataclass
class MultispeciesVariableProjectionMultistartResult:
    best_result: MultispeciesVariableProjectionResult
    best_index: int
    successful_results: list[MultispeciesVariableProjectionResult]
    starting_parameter_sets: list[dict[str, float]]
    comparison_table: pd.DataFrame
    failures: list[MultispeciesVariableProjectionStartFailure]


def _sample_parameter_specs(
    parameter_specs: list[ParameterSpec],
    rng: np.random.Generator,
    *,
    log_uniform: bool = True,
) -> tuple[list[ParameterSpec], dict[str, float]]:
    sampled_specs = []
    sampled_values = {}

    for spec in parameter_specs:
        lower = float(spec.lower_bound)
        upper = float(spec.upper_bound)

        if log_uniform and lower > 0 and upper > 0:
            value = float(
                np.exp(
                    rng.uniform(
                        np.log(lower),
                        np.log(upper),
                    )
                )
            )
        else:
            value = float(rng.uniform(lower, upper))

        sampled_values[spec.name] = value

        sampled_specs.append(
            ParameterSpec(
                name=spec.name,
                initial_guess=value,
                lower_bound=lower,
                upper_bound=upper,
            )
        )

    return sampled_specs, sampled_values


def fit_global_observable_model_multispecies_variable_projection_multistart(
    *,
    model: ModelSpec,
    dataset: Dataset,
    parameter_specs: list[ParameterSpec],
    initial_condition_specs: list[InitialConditionSpec],
    observed_species: list[str],
    settings: FitSettings,
    signal_columns: list[str] | None = None,
    fit_offset: bool = True,
    backend: str = "numpy",
    method: str = "LSODA",
    n_starts: int = 10,
    random_seed: int | None = None,
    sort_by: str = "bic",
    log_uniform: bool = True,
    show_progress: bool = True,
    engine_name: str = "reference",
) -> MultispeciesVariableProjectionMultistartResult:
    if n_starts < 1:
        raise ValueError("n_starts must be at least 1.")

    sort_by = sort_by.lower()

    if sort_by not in {"rss", "rmse", "aic", "bic"}:
        raise ValueError("sort_by must be one of: rss, rmse, aic, bic")

    if signal_columns is None:
        signal_columns = dataset.signal_columns

    rng = np.random.default_rng(random_seed)

    successful_results = []
    starting_parameter_sets = []
    failures = []
    rows = []

    for start_index in range(n_starts):
        if show_progress:
            print(
                f"Multispecies variable projection multistart: "
                f"{start_index + 1}/{n_starts}"
            )

        try:
            start_specs, sampled_values = _sample_parameter_specs(
                parameter_specs,
                rng,
                log_uniform=log_uniform,
            )

            result = fit_global_observable_model_multispecies_variable_projection(
                model=model,
                dataset=dataset,
                parameter_specs=start_specs,
                initial_condition_specs=initial_condition_specs,
                observed_species=observed_species,
                settings=settings,
                signal_columns=signal_columns,
                fit_offset=fit_offset,
                backend=backend,
                method=method,
                engine_name=engine_name,
            )

            successful_results.append(result)
            starting_parameter_sets.append(sampled_values)

            rows.append(
                {
                    "start_index": start_index,
                    "success": bool(result.success),
                    "rss": float(result.statistics["rss"]),
                    "rmse": float(result.statistics["rmse"]),
                    "aic": float(result.statistics["aic"]),
                    "bic": float(result.statistics["bic"]),
                    "message": result.message,
                    **{
                        f"start_{name}": value
                        for name, value in sampled_values.items()
                    },
                    **{
                        f"fit_{name}": value
                        for name, value in result.fitted_parameters.items()
                    },
                }
            )

        except Exception as exc:
            failures.append(
                MultispeciesVariableProjectionStartFailure(
                    start_index=start_index,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    traceback=traceback.format_exc(),
                )
            )

            rows.append(
                {
                    "start_index": start_index,
                    "success": False,
                    "rss": np.inf,
                    "rmse": np.inf,
                    "aic": np.inf,
                    "bic": np.inf,
                    "message": str(exc),
                }
            )

    if not successful_results:
        raise RuntimeError("All multispecies variable projection starts failed.")

    comparison_table = pd.DataFrame(rows).sort_values(
        by=[sort_by, "start_index"],
        ascending=[True, True],
    ).reset_index(drop=True)

    best_start_index = int(comparison_table.iloc[0]["start_index"])

    best_result = successful_results[
        [
            int(row["start_index"])
            for row in rows
            if row["success"]
        ].index(best_start_index)
    ]

    return MultispeciesVariableProjectionMultistartResult(
        best_result=best_result,
        best_index=best_start_index,
        successful_results=successful_results,
        starting_parameter_sets=starting_parameter_sets,
        comparison_table=comparison_table,
        failures=failures,
    )


def export_multispecies_variable_projection_multistart_summary(
    *,
    result: MultispeciesVariableProjectionMultistartResult,
    output_dir: str | Path,
    export_best_fit: bool = True,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    written_files = {}

    comparison_path = output_path / "multispecies_variable_projection_multistart.csv"
    result.comparison_table.to_csv(comparison_path, index=False)
    written_files["comparison"] = comparison_path

    metadata_path = output_path / "multispecies_variable_projection_multistart_summary.json"

    metadata = {
        "best_index": result.best_index,
        "n_successful_starts": len(result.successful_results),
        "n_failed_starts": len(result.failures),
        "best_statistics": result.best_result.statistics,
        "best_fitted_parameters": result.best_result.fitted_parameters,
    }

    with metadata_path.open("w") as handle:
        json.dump(metadata, handle, indent=2)

    written_files["summary"] = metadata_path

    if result.failures:
        failures_path = output_path / "multispecies_variable_projection_multistart_failures.csv"

        pd.DataFrame(
            [
                {
                    "start_index": failure.start_index,
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

    return written_files
