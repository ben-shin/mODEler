from __future__ import annotations

import json
import traceback
from dataclasses import dataclass
from pathlib import Path

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
    fit_global_observable_model_variable_projection,
)
from odefit.model.model_spec import ModelSpec
from odefit.plotting.profile_likelihood_plots import plot_profile_likelihoods


@dataclass
class ProfileLikelihoodFailure:
    parameter: str
    fixed_value: float
    error_type: str
    error_message: str
    traceback: str


@dataclass
class ProfileLikelihoodResult:
    original_result: object
    profile_table: pd.DataFrame
    failures: list[ProfileLikelihoodFailure]


def _replace_parameter_spec(
    parameter_specs: list[ParameterSpec],
    parameter_name: str,
    fixed_value: float,
) -> list[ParameterSpec]:
    new_specs = []

    for spec in parameter_specs:
        if spec.name == parameter_name:
            new_specs.append(
                ParameterSpec(
                    name=spec.name,
                    initial_guess=float(fixed_value),
                    lower_bound=float(fixed_value),
                    upper_bound=float(fixed_value),
                )
            )
        else:
            new_specs.append(spec)

    return new_specs


def _make_profile_grid(
    center: float,
    lower: float,
    upper: float,
    n_points: int,
    log_space: bool,
) -> np.ndarray:
    if n_points < 3:
        raise ValueError("n_points must be at least 3.")

    if log_space:
        if lower <= 0 or upper <= 0:
            raise ValueError("Log-space profile grid requires positive bounds.")
        return np.geomspace(lower, upper, n_points)

    return np.linspace(lower, upper, n_points)


def fit_variable_projection_profile_likelihood(
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
    profile_parameters: list[str] | None = None,
    n_points: int = 15,
    span_factor: float = 10.0,
    log_space: bool = True,
    show_progress: bool = True,
    engine_name: str = "reference",
) -> ProfileLikelihoodResult:
    if signal_columns is None:
        signal_columns = dataset.signal_columns

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
        engine_name=engine_name,
    )

    fitted = original_result.fitted_parameters

    if profile_parameters is None:
        profile_parameters = list(fitted)

    specs_by_name = {spec.name: spec for spec in parameter_specs}

    rows = []
    failures: list[ProfileLikelihoodFailure] = []

    base_rss = float(original_result.statistics["rss"])

    for parameter in profile_parameters:
        if parameter not in fitted:
            raise ValueError(f"Unknown fitted parameter for profile: {parameter}")

        spec = specs_by_name[parameter]
        center = float(fitted[parameter])

        lower = max(float(spec.lower_bound), center / span_factor)
        upper = min(float(spec.upper_bound), center * span_factor)

        if lower == upper:
            lower = float(spec.lower_bound)
            upper = float(spec.upper_bound)

        grid = _make_profile_grid(
            center=center,
            lower=lower,
            upper=upper,
            n_points=n_points,
            log_space=log_space,
        )

        for index, fixed_value in enumerate(grid):
            if show_progress:
                print(
                    f"Profile likelihood {parameter}: "
                    f"{index + 1}/{len(grid)} fixed={fixed_value:g}"
                )

            try:
                profiled_specs = _replace_parameter_spec(
                    parameter_specs=parameter_specs,
                    parameter_name=parameter,
                    fixed_value=float(fixed_value),
                )

                result = fit_global_observable_model_variable_projection(
                    model=model,
                    dataset=dataset,
                    parameter_specs=profiled_specs,
                    initial_condition_specs=initial_condition_specs,
                    observed_species=observed_species,
                    settings=settings,
                    signal_columns=signal_columns,
                    fit_scale=fit_scale,
                    fit_offset=fit_offset,
                    backend=backend,
                    method=method,
                    engine_name=engine_name,
                )

                rss = float(result.statistics["rss"])

                rows.append(
                    {
                        "parameter": parameter,
                        "fixed_value": float(fixed_value),
                        "rss": rss,
                        "delta_rss": rss - base_rss,
                        "rmse": float(result.statistics.get("rmse", np.nan)),
                        "aic": float(result.statistics.get("aic", np.nan)),
                        "bic": float(result.statistics.get("bic", np.nan)),
                        "success": bool(result.success),
                        "message": result.message,
                    }
                )

            except Exception as exc:
                failures.append(
                    ProfileLikelihoodFailure(
                        parameter=parameter,
                        fixed_value=float(fixed_value),
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                        traceback=traceback.format_exc(),
                    )
                )

                rows.append(
                    {
                        "parameter": parameter,
                        "fixed_value": float(fixed_value),
                        "rss": np.inf,
                        "delta_rss": np.inf,
                        "rmse": np.inf,
                        "aic": np.inf,
                        "bic": np.inf,
                        "success": False,
                        "message": str(exc),
                    }
                )

    return ProfileLikelihoodResult(
        original_result=original_result,
        profile_table=pd.DataFrame(rows),
        failures=failures,
    )


def export_profile_likelihood_result(
    *,
    result: ProfileLikelihoodResult,
    output_dir: str | Path,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    written_files: dict[str, Path] = {}

    table_path = output_path / "profile_likelihood.csv"
    result.profile_table.to_csv(table_path, index=False)
    written_files["profile_likelihood"] = table_path

    metadata = {
        "n_profile_points": int(len(result.profile_table)),
        "n_failures": int(len(result.failures)),
        "original_statistics": result.original_result.statistics,
        "original_fitted_parameters": result.original_result.fitted_parameters,
    }

    metadata_path = output_path / "profile_likelihood_metadata.json"
    with metadata_path.open("w") as handle:
        json.dump(metadata, handle, indent=2)

    written_files["metadata"] = metadata_path

    if result.failures:
        failures_path = output_path / "profile_likelihood_failures.csv"
        pd.DataFrame(
            [
                {
                    "parameter": failure.parameter,
                    "fixed_value": failure.fixed_value,
                    "error_type": failure.error_type,
                    "error_message": failure.error_message,
                    "traceback": failure.traceback,
                }
                for failure in result.failures
            ]
        ).to_csv(failures_path, index=False)

        written_files["failures"] = failures_path

    plot_files = plot_profile_likelihoods(
        profile_table=result.profile_table,
        output_dir=output_path / "plots",
    )

    for name, path in plot_files.items():
        written_files[f"plot_{name}"] = path

        identifiability_report = build_identifiability_report(
            profile_likelihood_table=result.profile_table,
            fitted_parameters=result.original_result.fitted_parameters,
            parameter_bounds=None,
        )

        identifiability_files = export_identifiability_report(
            report=identifiability_report,
            output_dir=output_path,
        )

        for name, path in identifiability_files.items():
            written_files[name] = path

    return written_files
