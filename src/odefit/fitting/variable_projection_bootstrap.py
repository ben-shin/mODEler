from __future__ import annotations

import json
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from odefit.data.dataset import Dataset
from odefit.fitting.fit_settings import FitSettings
from odefit.fitting.identifiability import (
    build_identifiability_report,
    export_identifiability_report,
)
from odefit.fitting.initial_condition_spec import InitialConditionSpec
from odefit.fitting.parameter_spec import ParameterSpec
from odefit.fitting.variable_projection import (
    export_variable_projection_fit,
    fit_global_observable_model_variable_projection,
)
from odefit.model.model_spec import ModelSpec
from odefit.plotting.bootstrap_plots import (
    plot_bootstrap_parameter_histograms,
    plot_bootstrap_parameter_pairs,
    plot_bootstrap_prediction_bands,
)


@dataclass
class VariableProjectionBootstrapFailure:
    bootstrap_index: int
    error_type: str
    error_message: str
    traceback: str


@dataclass
class VariableProjectionBootstrapResult:
    original_result: Any
    bootstrap_results: list[Any]
    parameter_samples: pd.DataFrame
    summary_table: pd.DataFrame
    failures: list[VariableProjectionBootstrapFailure]


def _build_resampled_dataset(
    *,
    original_dataset: Dataset,
    original_result: Any,
    signal_columns: list[str],
    rng: np.random.Generator,
) -> Dataset:
    """
    Residual bootstrap dataset.

    Uses:
        bootstrapped_y = fitted_y + resampled_residuals

    Residuals are resampled independently within each observable column.
    """

    raw = original_dataset.raw_dataframe.copy()

    fitted = original_result.predicted_dataframe
    residuals = original_result.residuals_dataframe

    boot = raw.copy()

    for column in signal_columns:
        fitted_values = np.asarray(
            fitted[column],
            dtype=float,
        )

        residual_values = np.asarray(
            residuals[column],
            dtype=float,
        )
        finite = np.isfinite(residual_values)

        if not finite.any():
            raise ValueError(
                f"No finite residuals available for bootstrap column: {column}"
            )

        sampled_residuals = rng.choice(
            residual_values[finite],
            size=len(residual_values),
            replace=True,
        )

        boot[column] = fitted_values + sampled_residuals

    return Dataset(
        raw_dataframe=boot,
        time_column=original_dataset.time_column,
        signal_columns=signal_columns,
    )


def _summarize_parameter_samples(
    parameter_samples: pd.DataFrame,
    confidence_level: float,
) -> pd.DataFrame:
    alpha = 1.0 - confidence_level
    lower_q = alpha / 2.0
    upper_q = 1.0 - alpha / 2.0

    rows = []

    parameter_columns = [
        column for column in parameter_samples.columns if column != "bootstrap_index"
    ]

    for parameter_name in parameter_columns:
        values = parameter_samples[parameter_name].dropna()

        rows.append(
            {
                "parameter": parameter_name,
                "n": int(values.shape[0]),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)),
                "median": float(values.median()),
                "ci_lower": float(values.quantile(lower_q)),
                "ci_upper": float(values.quantile(upper_q)),
                "confidence_level": confidence_level,
            }
        )

    return pd.DataFrame(rows)


def bootstrap_global_observable_variable_projection_fit(
    *,
    model: ModelSpec,
    dataset: Dataset,
    parameter_specs: list[ParameterSpec],
    initial_condition_specs: list[InitialConditionSpec],
    observed_species: str,
    settings: FitSettings,
    signal_columns: list[str] | None = None,
    fit_scale: bool = True,
    fit_offset: bool = True,
    backend: str = "numpy",
    method: str = "LSODA",
    n_bootstrap: int = 100,
    random_seed: int | None = None,
    confidence_level: float = 0.95,
    refit_original: bool = True,
    original_result: Any | None = None,
    show_progress: bool = True,
) -> VariableProjectionBootstrapResult:
    """
    Residual bootstrap for variable-projection global observable fitting.

    Procedure:
        1. Fit original dataset unless original_result is provided.
        2. Build bootstrap datasets by resampling residuals.
        3. Refit each bootstrapped dataset.
        4. Summarize kinetic parameter uncertainty.

    This estimates uncertainty in the nonlinear kinetic parameters.
    """

    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be at least 1.")

    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1.")

    if signal_columns is None:
        signal_columns = dataset.signal_columns

    rng = np.random.default_rng(random_seed)

    if original_result is None:
        if not refit_original:
            raise ValueError(
                "original_result must be supplied when refit_original=False."
            )

        original_result = fit_global_observable_model_variable_projection(
            model=model,
            dataset=dataset,
            parameter_specs=parameter_specs,
            initial_condition_specs=initial_condition_specs,
            observed_species=observed_species,
            settings=settings,
            signal_columns=signal_columns,
            fit_scale=fit_scale,
            fit_offset=fit_offset,
            backend=backend,
            method=method,
        )

    bootstrap_results: list[Any] = []
    failures: list[VariableProjectionBootstrapFailure] = []
    parameter_rows = []

    for bootstrap_index in range(n_bootstrap):
        if show_progress:
            print(f"Variable projection bootstrap: {bootstrap_index + 1}/{n_bootstrap}")

        try:
            bootstrap_dataset = _build_resampled_dataset(
                original_dataset=dataset,
                original_result=original_result,
                signal_columns=signal_columns,
                rng=rng,
            )

            fit_result = fit_global_observable_model_variable_projection(
                model=model,
                dataset=bootstrap_dataset,
                parameter_specs=parameter_specs,
                initial_condition_specs=initial_condition_specs,
                observed_species=observed_species,
                settings=settings,
                signal_columns=signal_columns,
                fit_scale=fit_scale,
                fit_offset=fit_offset,
                backend=backend,
                method=method,
            )

            bootstrap_results.append(fit_result)

            row = {"bootstrap_index": bootstrap_index}
            row.update(fit_result.fitted_parameters)
            parameter_rows.append(row)

        except Exception as exc:
            failures.append(
                VariableProjectionBootstrapFailure(
                    bootstrap_index=bootstrap_index,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    traceback=traceback.format_exc(),
                )
            )

    parameter_samples = pd.DataFrame(parameter_rows)

    if parameter_samples.empty:
        raise RuntimeError("All bootstrap fits failed.")

    summary_table = _summarize_parameter_samples(
        parameter_samples=parameter_samples,
        confidence_level=confidence_level,
    )

    return VariableProjectionBootstrapResult(
        original_result=original_result,
        bootstrap_results=bootstrap_results,
        parameter_samples=parameter_samples,
        summary_table=summary_table,
        failures=failures,
    )


def export_variable_projection_bootstrap_result(
    *,
    result: VariableProjectionBootstrapResult,
    output_dir: str | Path,
    export_original_fit: bool = True,
    include_plots: bool = True,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    written_files: dict[str, Path] = {}

    samples_path = output_path / "bootstrap_parameter_samples.csv"
    result.parameter_samples.to_csv(samples_path, index=False)
    written_files["bootstrap_parameter_samples"] = samples_path

    summary_path = output_path / "bootstrap_parameter_summary.csv"
    result.summary_table.to_csv(summary_path, index=False)
    written_files["bootstrap_parameter_summary"] = summary_path

    metadata = {
        "n_requested_bootstrap": (len(result.bootstrap_results) + len(result.failures)),
        "n_successful_bootstrap": len(result.bootstrap_results),
        "n_failed_bootstrap": len(result.failures),
    }

    identifiability_report = build_identifiability_report(
        bootstrap_summary_table=result.summary_table,
        fitted_parameters=result.original_result.fitted_parameters,
        parameter_bounds=None,
        n_bootstrap_requested=(len(result.bootstrap_results) + len(result.failures)),
        n_bootstrap_failed=len(result.failures),
    )

    identifiability_files = export_identifiability_report(
        report=identifiability_report,
        output_dir=output_path,
    )

    for name, path in identifiability_files.items():
        written_files[name] = path

    metadata_path = output_path / "bootstrap_metadata.json"
    with metadata_path.open("w") as handle:
        json.dump(metadata, handle, indent=2)
    written_files["bootstrap_metadata"] = metadata_path

    if result.failures:
        failures_path = output_path / "bootstrap_failures.csv"
        pd.DataFrame(
            [
                {
                    "bootstrap_index": failure.bootstrap_index,
                    "error_type": failure.error_type,
                    "error_message": failure.error_message,
                    "traceback": failure.traceback,
                }
                for failure in result.failures
            ]
        ).to_csv(failures_path, index=False)
        written_files["bootstrap_failures"] = failures_path

    if export_original_fit:
        original_dir = output_path / "original_fit"

        original_files = export_variable_projection_fit(
            result=result.original_result,
            output_dir=original_dir,
        )

        for name, path in original_files.items():
            written_files[f"original_fit_{name}"] = Path(path)

    if include_plots:
        plots_dir = output_path / "plots"

        histogram_files = plot_bootstrap_parameter_histograms(
            parameter_samples=result.parameter_samples,
            output_dir=plots_dir,
        )

        for name, path in histogram_files.items():
            written_files[f"plot_{name}"] = path

        pair_files = plot_bootstrap_parameter_pairs(
            parameter_samples=result.parameter_samples,
            output_dir=plots_dir,
        )

        for name, path in pair_files.items():
            written_files[f"plot_{name}"] = path

        bootstrap_prediction_dataframes = [
            bootstrap_result.predicted_dataframe
            for bootstrap_result in result.bootstrap_results
            if hasattr(bootstrap_result, "predicted_dataframe")
        ]

        if bootstrap_prediction_dataframes:
            signal_columns = [
                column
                for column in result.original_result.predicted_dataframe.columns
                if column != result.original_result.predicted_dataframe.columns[0]
            ]

            prediction_files = plot_bootstrap_prediction_bands(
                original_dataframe=result.original_result.predicted_dataframe,
                bootstrap_prediction_dataframes=bootstrap_prediction_dataframes,
                time_column=result.original_result.predicted_dataframe.columns[0],
                signal_columns=signal_columns,
                output_dir=plots_dir,
            )

            for name, path in prediction_files.items():
                written_files[f"plot_{name}"] = path

    return written_files
