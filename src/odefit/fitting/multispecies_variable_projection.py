from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import numpy as np
import pandas as pd

from odefit.data.dataset import Dataset
from odefit.engines.base import BackendEngineBundle
from odefit.fitting.engine_helpers import (
    engine_least_squares,
    engine_project_multispecies,
    engine_solve_to_dataframe,
    resolve_engine_bundle,
)
from odefit.fitting.fit_settings import FitSettings
from odefit.fitting.initial_condition_spec import InitialConditionSpec
from odefit.fitting.parameter_spec import ParameterSpec
from odefit.model.model_spec import ModelSpec


@dataclass
class MultispeciesProjectionObservableFit:
    signal_column: str
    coefficients: dict[str, float]
    offset: float
    rss: float


@dataclass
class MultispeciesVariableProjectionResult:
    success: bool
    message: str
    fitted_parameters: dict[str, float]
    fitted_initial_conditions: dict[str, float]
    observable_table: pd.DataFrame
    simulation_dataframe: pd.DataFrame
    predicted_dataframe: pd.DataFrame
    residuals_dataframe: pd.DataFrame
    residual_vector: np.ndarray
    statistics: dict[str, float]
    optimizer_result: Any


def _parameter_vector_from_specs(
    parameter_specs: list[ParameterSpec],
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    names = [spec.name for spec in parameter_specs]

    x0 = np.asarray(
        [spec.initial_guess for spec in parameter_specs],
        dtype=float,
    )

    lower = np.asarray(
        [spec.lower_bound for spec in parameter_specs],
        dtype=float,
    )

    upper = np.asarray(
        [spec.upper_bound for spec in parameter_specs],
        dtype=float,
    )

    return names, x0, lower, upper


def _initial_conditions_from_specs(
    initial_condition_specs: list[InitialConditionSpec],
) -> dict[str, float]:
    initial_conditions = {}

    for spec in initial_condition_specs:
        if spec.fixed:
            initial_conditions[spec.species] = float(spec.fixed_value)
        else:
            initial_conditions[spec.species] = float(spec.initial_guess)

    return initial_conditions


def solve_multispecies_observable_projection(
    *,
    signal: np.ndarray,
    species_matrix: np.ndarray,
    species_names: list[str],
    fit_offset: bool = True,
) -> tuple[dict[str, float], float, np.ndarray, np.ndarray, float]:
    """
    Solve:

        y = a1*S1 + a2*S2 + ... + offset

    by linear least squares.

    Returns:
        coefficients
        offset
        predicted
        residuals
        rss
    """

    y = np.asarray(signal, dtype=float)
    X = np.asarray(species_matrix, dtype=float)

    finite = np.isfinite(y) & np.all(np.isfinite(X), axis=1)

    if not finite.any():
        raise ValueError("No finite data points available for projection.")

    X_fit = X[finite]
    y_fit = y[finite]

    if fit_offset:
        design = np.column_stack(
            [
                X_fit,
                np.ones(len(X_fit)),
            ]
        )
    else:
        design = X_fit

    beta, *_ = np.linalg.lstsq(
        design,
        y_fit,
        rcond=None,
    )

    if fit_offset:
        coefficient_values = beta[:-1]
        offset = float(beta[-1])
    else:
        coefficient_values = beta
        offset = 0.0

    coefficients = {
        species: float(value)
        for species, value in zip(species_names, coefficient_values)
    }

    predicted = X @ coefficient_values + offset
    residuals = y - predicted

    rss = float(
        np.nansum(
            residuals[finite] ** 2,
        )
    )

    return coefficients, offset, predicted, residuals, rss


def fit_global_observable_model_multispecies_variable_projection(
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
    engine_name: str = "reference",
    engine_bundle: BackendEngineBundle | None = None,
) -> MultispeciesVariableProjectionResult:
    """
    Multi-species variable projection.

    Nonlinear optimizer fits kinetic parameters.
    For each observable column, linear coefficients for selected species
    are solved analytically using the configured projection engine.

    Model:
        signal_i(t) = c_i1*S1(t) + c_i2*S2(t) + ... + offset_i
    """

    if backend != "numpy":
        raise ValueError("Only backend='numpy' is currently supported.")

    engine_bundle = resolve_engine_bundle(
        engine_name=engine_name,
        engine_bundle=engine_bundle,
    )

    if signal_columns is None:
        signal_columns = dataset.signal_columns

    if not observed_species:
        raise ValueError("observed_species must contain at least one species.")

    missing_species = [
        species for species in observed_species if species not in model.species
    ]

    if missing_species:
        raise ValueError(
            "Observed species not present in model: "
            + ", ".join(missing_species)
        )

    parameter_names, x0, lower, upper = _parameter_vector_from_specs(
        parameter_specs,
    )

    initial_conditions = _initial_conditions_from_specs(
        initial_condition_specs,
    )

    timepoints = dataset.time_values

    observed_matrix = dataset.raw_dataframe[signal_columns].to_numpy(
        dtype=float,
    )

    last_context: dict[str, Any] = {}

    def residual_function(parameter_vector: np.ndarray) -> np.ndarray:
        parameters = {
            name: float(value)
            for name, value in zip(parameter_names, parameter_vector)
        }

        simulation_dataframe = engine_solve_to_dataframe(
            engine_bundle=engine_bundle,
            model=model,
            parameters=parameters,
            initial_conditions=initial_conditions,
            timepoints=timepoints,
            settings=(
                settings.to_simulation_settings()
                if hasattr(settings, "to_simulation_settings")
                else None
            ),
        )

        species_matrix = simulation_dataframe[observed_species].to_numpy(
            dtype=float,
        )

        residual_parts = []
        observable_rows = []
        predicted_dataframe = pd.DataFrame({"time": timepoints})
        residuals_dataframe = pd.DataFrame({"time": timepoints})

        for column_index, signal_column in enumerate(signal_columns):
            signal = observed_matrix[:, column_index]

            projection = engine_project_multispecies(
                engine_bundle=engine_bundle,
                observed_values=signal,
                species_matrix=species_matrix,
                species_names=observed_species,
                fit_offset=fit_offset,
            )

            coefficients = projection.coefficients
            offset = projection.offset
            predicted = projection.predicted
            residuals = projection.residuals
            rss = projection.rss

            finite = np.isfinite(residuals)

            residual_parts.append(residuals[finite])

            row = {
                "signal_column": signal_column,
                "offset": offset,
                "rss": rss,
            }

            for species, coefficient in coefficients.items():
                row[f"coefficient_{species}"] = coefficient

            observable_rows.append(row)

            predicted_dataframe[signal_column] = predicted
            residuals_dataframe[signal_column] = residuals

        residual_vector = np.concatenate(residual_parts)

        last_context.clear()
        last_context.update(
            {
                "parameters": parameters,
                "simulation_dataframe": simulation_dataframe,
                "observable_table": pd.DataFrame(observable_rows),
                "predicted_dataframe": predicted_dataframe,
                "residuals_dataframe": residuals_dataframe,
                "residual_vector": residual_vector,
            }
        )

        return residual_vector

    optimizer_result = engine_least_squares(
        engine_bundle=engine_bundle,
        residual_function=residual_function,
        x0=x0,
        bounds=(lower, upper),
        method=settings.method,
        loss=settings.loss,
        max_nfev=settings.max_nfev,
    )

    residual_vector = residual_function(optimizer_result.x)

    rss = float(np.sum(residual_vector**2))
    n_points = int(len(residual_vector))
    n_kinetic_parameters = len(parameter_names)

    rmse = float(np.sqrt(rss / n_points))

    aic = float(
        n_points * np.log(max(rss / n_points, np.finfo(float).tiny))
        + 2 * n_kinetic_parameters
    )

    bic = float(
        n_points * np.log(max(rss / n_points, np.finfo(float).tiny))
        + n_kinetic_parameters * np.log(n_points)
    )

    return MultispeciesVariableProjectionResult(
        success=bool(optimizer_result.success),
        message=str(optimizer_result.message),
        fitted_parameters={
            name: float(value)
            for name, value in zip(parameter_names, optimizer_result.x)
        },
        fitted_initial_conditions=initial_conditions,
        observable_table=last_context["observable_table"],
        simulation_dataframe=last_context["simulation_dataframe"],
        predicted_dataframe=last_context["predicted_dataframe"],
        residuals_dataframe=last_context["residuals_dataframe"],
        residual_vector=last_context["residual_vector"],
        statistics={
            "rss": rss,
            "rmse": rmse,
            "aic": aic,
            "bic": bic,
            "n_points": n_points,
            "n_kinetic_parameters": n_kinetic_parameters,
            "n_observable_columns": len(signal_columns),
            "n_projected_species": len(observed_species),
        },
        optimizer_result=optimizer_result,
    )


def export_multispecies_variable_projection_fit(
    *,
    result: MultispeciesVariableProjectionResult,
    output_dir,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    written_files = {}

    observable_path = output_path / "multispecies_observable_table.csv"
    result.observable_table.to_csv(observable_path, index=False)
    written_files["observable_table"] = observable_path

    simulation_path = output_path / "simulation.csv"
    result.simulation_dataframe.to_csv(simulation_path, index=False)
    written_files["simulation"] = simulation_path

    predicted_path = output_path / "predicted.csv"
    result.predicted_dataframe.to_csv(predicted_path, index=False)
    written_files["predicted"] = predicted_path

    residuals_path = output_path / "residuals.csv"
    result.residuals_dataframe.to_csv(residuals_path, index=False)
    written_files["residuals"] = residuals_path

    summary_path = output_path / "summary.json"

    summary = {
        "success": result.success,
        "message": result.message,
        "fitted_parameters": result.fitted_parameters,
        "fitted_initial_conditions": result.fitted_initial_conditions,
        "statistics": result.statistics,
    }

    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2)

    written_files["summary"] = summary_path

    return written_files
