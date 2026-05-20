from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import traceback

import numpy as np
import pandas as pd

from odefit.data.dataset import Dataset
from odefit.fitting.fit_settings import FitSettings
from odefit.fitting.identifiability import (
    build_identifiability_report,
    export_identifiability_report,
)
from odefit.fitting.initial_condition_spec import InitialConditionSpec
from odefit.fitting.multispecies_variable_projection import (
    MultispeciesVariableProjectionResult,
    export_multispecies_variable_projection_fit,
    fit_global_observable_model_multispecies_variable_projection,
)
from odefit.fitting.parameter_spec import ParameterSpec
from odefit.model.model_spec import ModelSpec
from odefit.plotting.bootstrap_plots import (
    plot_bootstrap_parameter_histograms,
    plot_bootstrap_parameter_pairs,
    plot_bootstrap_prediction_bands,
)


@dataclass
class MultispeciesVariableProjectionBootstrapFailure:
    bootstrap_index: int
    error_type: str
    error_message: str
    traceback: str


@dataclass
class MultispeciesVariableProjectionBootstrapResult:
    original_result: MultispeciesVariableProjectionResult
    bootstrap_results: list[MultispeciesVariableProjectionResult]
    parameter_samples: pd.DataFrame
    summary_table: pd.DataFrame
    failures: list[MultispeciesVariableProjectionBootstrapFailure]


def _build_resampled_dataset(
    *,
    original_dataset: Dataset,
    original_result: MultispeciesVariableProjectionResult,
    signal_columns: list[str],
    rng: np.random.Generator,
) -> Dataset:
    raw = original_dataset.raw_dataframe.copy()
    boot = raw.copy()

    fitted = original_result.predicted_dataframe
    residuals = original_result.residuals_dataframe

    for column in signal_columns:
        fitted_values = np.asarray(fitted[column], dtype=float)
        residual_values = np.asarray(residuals[column], dtype=float)

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


def _fit_one_bootstrap_worker(payload: dict):
    bootstrap_index = payload["bootstrap_index"]
    rng = np.random.default_rng(payload["random_seed"])

    bootstrap_dataset = _build_resampled_dataset(
        original_dataset=payload["dataset"],
        original_result=payload["original_result"],
        signal_columns=payload["signal_columns"],
        rng=rng,
    )

    fit_result = fit_global_observable_model_multispecies_variable_projection(
        model=payload["model"],
        dataset=bootstrap_dataset,
        parameter_specs=payload["parameter_specs"],
        initial_condition_specs=payload["initial_condition_specs"],
        observed_species=payload["observed_species"],
        settings=payload["settings"],
        signal_columns=payload["signal_columns"],
        fit_offset=payload["fit_offset"],
        backend=payload["backend"],
        method=payload["method"],
        engine_name=payload.get("engine_name", "reference"),
    )

    row = {"bootstrap_index": bootstrap_index}
    row.update(fit_result.fitted_parameters)

    return bootstrap_index, fit_result, row


def bootstrap_global_observable_multispecies_variable_projection_fit(
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
    n_bootstrap: int = 100,
    n_workers: int = 1,
    random_seed: int | None = None,
    confidence_level: float = 0.95,
    original_result: MultispeciesVariableProjectionResult | None = None,
    show_progress: bool = True,
    engine_name: str = "reference",
) -> MultispeciesVariableProjectionBootstrapResult:
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be at least 1.")

    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1.")

    if signal_columns is None:
        signal_columns = dataset.signal_columns

    rng = np.random.default_rng(random_seed)

    if original_result is None:
        original_result = fit_global_observable_model_multispecies_variable_projection(
            model=model,
            dataset=dataset,
            parameter_specs=parameter_specs,
            initial_condition_specs=initial_condition_specs,
            observed_species=observed_species,
            settings=settings,
            signal_columns=signal_columns,
            fit_offset=fit_offset,
            backend=backend,
            method=method,
            engine_name=engine_name,
        )

    bootstrap_results: list[MultispeciesVariableProjectionResult] = []
    failures: list[MultispeciesVariableProjectionBootstrapFailure] = []
    parameter_rows = []

    seeds = rng.integers(
        low=0,
        high=np.iinfo(np.int32).max,
        size=n_bootstrap,
    )

    if n_workers is None or int(n_workers) <= 1:
        for bootstrap_index in range(n_bootstrap):
            if show_progress:
                print(
                    f"Multispecies variable projection bootstrap: "
                    f"{bootstrap_index + 1}/{n_bootstrap}"
                )

            try:
                bootstrap_dataset = _build_resampled_dataset(
                    original_dataset=dataset,
                    original_result=original_result,
                    signal_columns=signal_columns,
                    rng=np.random.default_rng(int(seeds[bootstrap_index])),
                )

                fit_result = fit_global_observable_model_multispecies_variable_projection(
                    model=model,
                    dataset=bootstrap_dataset,
                    parameter_specs=parameter_specs,
                    initial_condition_specs=initial_condition_specs,
                    observed_species=observed_species,
                    settings=settings,
                    signal_columns=signal_columns,
                    fit_offset=fit_offset,
                    backend=backend,
                    method=method,
                    engine_name=engine_name,
                )

                bootstrap_results.append(fit_result)

                row = {"bootstrap_index": bootstrap_index}
                row.update(fit_result.fitted_parameters)
                parameter_rows.append(row)

            except Exception as exc:
                failures.append(
                    MultispeciesVariableProjectionBootstrapFailure(
                        bootstrap_index=bootstrap_index,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                        traceback=traceback.format_exc(),
                    )
                )

    else:
        payloads = [
            {
                "bootstrap_index": bootstrap_index,
                "random_seed": int(seeds[bootstrap_index]),
                "model": model,
                "dataset": dataset,
                "original_result": original_result,
                "parameter_specs": parameter_specs,
                "initial_condition_specs": initial_condition_specs,
                "observed_species": observed_species,
                "settings": settings,
                "signal_columns": signal_columns,
                "fit_offset": fit_offset,
                "backend": backend,
                "method": method,
                "engine_name": engine_name,
            }
            for bootstrap_index in range(n_bootstrap)
        ]

        completed = 0

        with ProcessPoolExecutor(max_workers=int(n_workers)) as executor:
            futures = {
                executor.submit(_fit_one_bootstrap_worker, payload): payload[
                    "bootstrap_index"
                ]
                for payload in payloads
            }

            for future in as_completed(futures):
                bootstrap_index = futures[future]
                completed += 1

                if show_progress:
                    print(
                        f"Multispecies variable projection bootstrap: "
                        f"{completed}/{n_bootstrap}"
                    )

                try:
                    _, fit_result, row = future.result()
                    bootstrap_results.append(fit_result)
                    parameter_rows.append(row)

                except Exception as exc:
                    failures.append(
                        MultispeciesVariableProjectionBootstrapFailure(
                            bootstrap_index=bootstrap_index,
                            error_type=type(exc).__name__,
                            error_message=str(exc),
                            traceback=traceback.format_exc(),
                        )
                    )

    parameter_samples = pd.DataFrame(parameter_rows)

    if not parameter_samples.empty:
        parameter_samples = parameter_samples.sort_values(
            "bootstrap_index"
        ).reset_index(drop=True)

    if parameter_samples.empty:
        raise RuntimeError("All multispecies bootstrap fits failed.")

    summary_table = _summarize_parameter_samples(
        parameter_samples=parameter_samples,
        confidence_level=confidence_level,
    )

    return MultispeciesVariableProjectionBootstrapResult(
        original_result=original_result,
        bootstrap_results=bootstrap_results,
        parameter_samples=parameter_samples,
        summary_table=summary_table,
        failures=failures,
    )


def export_multispecies_variable_projection_bootstrap_result(
    *,
    result: MultispeciesVariableProjectionBootstrapResult,
    output_dir: str | Path,
    export_original_fit: bool = True,
    include_plots: bool = True,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    written_files: dict[str, Path] = {}

    samples_path = output_path / "multispecies_bootstrap_parameter_samples.csv"
    result.parameter_samples.to_csv(samples_path, index=False)
    written_files["bootstrap_parameter_samples"] = samples_path

    summary_path = output_path / "multispecies_bootstrap_parameter_summary.csv"
    result.summary_table.to_csv(summary_path, index=False)
    written_files["bootstrap_parameter_summary"] = summary_path

    metadata_path = output_path / "multispecies_bootstrap_metadata.json"

    metadata = {
        "n_requested_bootstrap": len(result.bootstrap_results) + len(result.failures),
        "n_successful_bootstrap": len(result.bootstrap_results),
        "n_failed_bootstrap": len(result.failures),
    }

    with metadata_path.open("w") as handle:
        json.dump(metadata, handle, indent=2)

    written_files["bootstrap_metadata"] = metadata_path

    if result.failures:
        failures_path = output_path / "multispecies_bootstrap_failures.csv"

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
        original_files = export_multispecies_variable_projection_fit(
            result=result.original_result,
            output_dir=output_path / "original_fit",
        )

        for name, path in original_files.items():
            written_files[f"original_fit_{name}"] = path

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

        prediction_files = plot_bootstrap_prediction_bands(
            original_dataframe=result.original_result.predicted_dataframe,
            bootstrap_prediction_dataframes=[
                bootstrap_result.predicted_dataframe
                for bootstrap_result in result.bootstrap_results
            ],
            time_column=result.original_result.predicted_dataframe.columns[0],
            signal_columns=[
                column
                for column in result.original_result.predicted_dataframe.columns
                if column != result.original_result.predicted_dataframe.columns[0]
            ],
            output_dir=plots_dir,
        )

        for name, path in prediction_files.items():
            written_files[f"plot_{name}"] = path

    identifiability_report = build_identifiability_report(
        bootstrap_summary_table=result.summary_table,
        fitted_parameters=result.original_result.fitted_parameters,
        parameter_bounds=None,
        n_bootstrap_requested=len(result.bootstrap_results) + len(result.failures),
        n_bootstrap_failed=len(result.failures),
    )

    identifiability_files = export_identifiability_report(
        report=identifiability_report,
        output_dir=output_path,
    )

    for name, path in identifiability_files.items():
        written_files[name] = path

    return written_files
