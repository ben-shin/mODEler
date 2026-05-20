from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from odefit.data.dataset import Dataset
from odefit.engines.base import BackendEngineBundle
from odefit.fitting.engine_helpers import (
    engine_least_squares,
    engine_project_single_species,
    engine_solve_to_dataframe,
    resolve_engine_bundle,
)
from odefit.fitting.fit_settings import FitSettings
from odefit.fitting.initial_condition_spec import InitialConditionSpec
from odefit.fitting.parameter_spec import ParameterSpec
from odefit.model.model_spec import ModelSpec
from odefit.performance.array_solve_ivp import (
    ArraySolveResult,
    solve_array_mass_action_model,
)


@dataclass
class LinearObservableProjectionResult:
    """
    Result from projecting observed signals onto one simulated species curve.

    For each data column, this solves:

        y_i(t) = scale_i * x(t) + offset_i

    where x(t) is the simulated species curve.
    """

    observable_table: pd.DataFrame
    predicted_dataframe: pd.DataFrame
    residuals_dataframe: pd.DataFrame
    residual_vector: np.ndarray
    rss: float
    n_observations: int
    n_linear_parameters: int


@dataclass
class VariableProjectionGlobalObservableFitResult:
    """
    Result from variable-projection global observable fitting.

    Nonlinear parameters:
        kinetic parameters, and later possibly initial conditions

    Linear parameters:
        per-observable scale/offset terms solved analytically
    """

    success: bool
    message: str
    fitted_parameters: dict[str, float]
    initial_parameters: dict[str, float]
    fitted_initial_conditions: dict[str, float]
    statistics: dict[str, float]
    observable_table: pd.DataFrame
    predicted_dataframe: pd.DataFrame
    residuals_dataframe: pd.DataFrame
    simulation_dataframe: pd.DataFrame
    residual_vector: np.ndarray
    simulation_result: ArraySolveResult
    cost: float
    nfev: int
    status: int
    optimality: float
    active_mask: np.ndarray


def solve_scale_offset(
    x_values: np.ndarray,
    y_values: np.ndarray,
    fit_scale: bool = True,
    fit_offset: bool = True,
) -> tuple[float, float]:
    """
    Solve y = scale * x + offset by linear least squares.

    Missing values are ignored.

    Supported modes:
    - fit scale and offset
    - fit scale only, offset fixed to zero
    - fit offset only, scale fixed to one
    - fit neither, scale fixed to one and offset fixed to zero
    """

    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)

    if x.shape != y.shape:
        raise ValueError(
            f"x_values and y_values must have the same shape: {x.shape} vs {y.shape}"
        )

    valid = np.isfinite(x) & np.isfinite(y)

    if valid.sum() < 2 and fit_scale and fit_offset:
        raise ValueError(
            "At least two finite points are required to fit scale and offset"
        )

    if valid.sum() < 1:
        raise ValueError("At least one finite point is required")

    x_valid = x[valid]
    y_valid = y[valid]

    if fit_scale and fit_offset:
        design = np.column_stack([x_valid, np.ones_like(x_valid)])
        coefficients, *_ = np.linalg.lstsq(
            design,
            y_valid,
            rcond=None,
        )
        scale = float(coefficients[0])
        offset = float(coefficients[1])
        return scale, offset

    if fit_scale and not fit_offset:
        denominator = float(np.dot(x_valid, x_valid))

        if denominator == 0.0:
            raise ValueError("Cannot fit scale because simulated curve is all zero")

        scale = float(np.dot(x_valid, y_valid) / denominator)
        return scale, 0.0

    if not fit_scale and fit_offset:
        offset = float(np.mean(y_valid - x_valid))
        return 1.0, offset

    return 1.0, 0.0


def project_observables_onto_species(
    timepoints: np.ndarray,
    simulated_species_values: np.ndarray,
    observed_dataframe: pd.DataFrame,
    signal_columns: list[str],
    fit_scale: bool = True,
    fit_offset: bool = True,
    signal_weights: dict[str, float] | None = None,
    engine_bundle: BackendEngineBundle | None = None,
) -> LinearObservableProjectionResult:
    """
    Project each observed signal column onto one simulated species curve.

    This analytically solves the best scale/offset for every signal column.
    """

    x = np.asarray(simulated_species_values, dtype=float)
    time_array = np.asarray(timepoints, dtype=float)

    if x.shape != time_array.shape:
        raise ValueError(
            f"simulated_species_values must have shape {time_array.shape}, got {x.shape}"
        )

    signal_weights = {} if signal_weights is None else signal_weights

    predicted_data: dict[str, Any] = {
        "time": time_array,
    }

    residual_data: dict[str, Any] = {
        "time": time_array,
    }

    observable_rows: list[dict[str, Any]] = []
    residual_blocks: list[np.ndarray] = []

    n_linear_parameters = 0

    for column in signal_columns:
        y = observed_dataframe[column].to_numpy(dtype=float)

        if engine_bundle is None:
            scale, offset = solve_scale_offset(
                x_values=x,
                y_values=y,
                fit_scale=fit_scale,
                fit_offset=fit_offset,
            )

            predicted = scale * x + offset
        else:
            projection = engine_project_single_species(
                engine_bundle=engine_bundle,
                observed_values=y,
                species_values=x,
                fit_scale=fit_scale,
                fit_offset=fit_offset,
            )

            scale = projection.scale
            offset = projection.offset
            predicted = projection.predicted

        residual = predicted - y

        weight = float(signal_weights.get(column, 1.0))
        weighted_residual = residual * weight

        predicted_data[column] = predicted
        residual_data[column] = residual
        residual_blocks.append(weighted_residual[np.isfinite(weighted_residual)])

        n_finite = int(np.isfinite(y).sum())
        column_rss = float(
            np.sum(weighted_residual[np.isfinite(weighted_residual)] ** 2)
        )

        if fit_scale:
            n_linear_parameters += 1

        if fit_offset:
            n_linear_parameters += 1

        observable_rows.append(
            {
                "data_column": column,
                "scale": scale,
                "offset": offset,
                "fit_scale": fit_scale,
                "fit_offset": fit_offset,
                "weight": weight,
                "n_observations": n_finite,
                "rss": column_rss,
                "rmse": float(np.sqrt(column_rss / max(n_finite, 1))),
            }
        )

    residual_vector = np.concatenate(residual_blocks)

    rss = float(np.sum(residual_vector**2))

    observable_table = pd.DataFrame(observable_rows)
    predicted_dataframe = pd.DataFrame(predicted_data)
    residuals_dataframe = pd.DataFrame(residual_data)

    return LinearObservableProjectionResult(
        observable_table=observable_table,
        predicted_dataframe=predicted_dataframe,
        residuals_dataframe=residuals_dataframe,
        residual_vector=residual_vector,
        rss=rss,
        n_observations=int(len(residual_vector)),
        n_linear_parameters=n_linear_parameters,
    )


def _build_initial_parameter_dict(
    parameter_specs: list[ParameterSpec],
) -> dict[str, float]:
    """
    Build initial full parameter dictionary from specs.
    """

    values: dict[str, float] = {}

    for spec in parameter_specs:
        if spec.tied_to is not None:
            raise NotImplementedError(
                "Variable projection prototype does not yet support tied parameters"
            )

        if spec.fixed:
            values[spec.name] = float(
                spec.fixed_value if spec.fixed_value is not None else spec.initial_guess
            )
        else:
            values[spec.name] = float(spec.initial_guess)

    return values


def _free_parameter_specs(
    parameter_specs: list[ParameterSpec],
) -> list[ParameterSpec]:
    """
    Return free nonlinear parameter specs.
    """

    return [spec for spec in parameter_specs if not spec.fixed]


def _parameter_vector_to_dict(
    parameter_specs: list[ParameterSpec],
    free_values: np.ndarray,
) -> dict[str, float]:
    """
    Convert optimizer vector to full parameter dictionary.
    """

    free_specs = _free_parameter_specs(parameter_specs)

    if len(free_specs) != len(free_values):
        raise ValueError("Free value vector length does not match free parameter specs")

    parameter_values = _build_initial_parameter_dict(parameter_specs)

    for spec, value in zip(free_specs, free_values):
        parameter_values[spec.name] = float(value)

    return parameter_values


def _build_fixed_initial_conditions(
    initial_condition_specs: list[InitialConditionSpec],
) -> dict[str, float]:
    """
    Build fixed initial condition dictionary.

    Step 10F prototype supports fixed initial conditions only.
    """

    initial_conditions: dict[str, float] = {}

    for spec in initial_condition_specs:
        if not spec.fixed:
            raise NotImplementedError(
                "Variable projection prototype currently supports fixed initial conditions only"
            )

        value = spec.fixed_value if spec.fixed_value is not None else spec.initial_guess
        initial_conditions[spec.species] = float(value)

    return initial_conditions


def _build_fit_statistics(
    rss: float,
    n_observations: int,
    n_nonlinear_parameters: int,
    n_linear_parameters: int,
) -> dict[str, float]:
    """
    Build fit statistics for variable projection.

    AIC/BIC count both nonlinear and projected linear parameters.
    """

    n_total_parameters = int(n_nonlinear_parameters + n_linear_parameters)
    dof = int(max(n_observations - n_total_parameters, 1))
    mse = float(rss / dof)
    rmse = float(np.sqrt(rss / max(n_observations, 1)))

    safe_rss = max(float(rss), np.finfo(float).tiny)
    safe_n = max(int(n_observations), 1)

    aic = float(safe_n * np.log(safe_rss / safe_n) + 2 * n_total_parameters)
    bic = float(
        safe_n * np.log(safe_rss / safe_n) + n_total_parameters * np.log(safe_n)
    )

    return {
        "rss": float(rss),
        "mse": mse,
        "rmse": rmse,
        "n_observations": float(n_observations),
        "n_nonlinear_parameters": float(n_nonlinear_parameters),
        "n_linear_parameters": float(n_linear_parameters),
        "n_total_parameters": float(n_total_parameters),
        "degrees_of_freedom": float(dof),
        "aic": aic,
        "bic": bic,
    }


def _simulation_to_dataframe(
    simulation_result: ArraySolveResult,
) -> pd.DataFrame:
    """
    Convert ArraySolveResult to dataframe.
    """

    data: dict[str, Any] = {
        "time": simulation_result.timepoints,
    }

    for species in simulation_result.species:
        data[species] = simulation_result.get_species_values(species)

    return pd.DataFrame(data)


def fit_global_observable_model_variable_projection(
    *,
    model: ModelSpec,
    dataset: Dataset,
    parameter_specs: list[ParameterSpec],
    initial_condition_specs: list[InitialConditionSpec],
    observed_species: str = "A",
    settings: FitSettings | None = None,
    signal_columns: list[str] | None = None,
    fit_scale: bool = True,
    fit_offset: bool = True,
    backend: str = "numpy",
    method: str = "LSODA",
    engine_name: str = "reference",
    engine_bundle: BackendEngineBundle | None = None,
) -> VariableProjectionGlobalObservableFitResult:
    """
    Fit a global observable model using variable projection.

    This prototype fits nonlinear kinetic parameters with scipy least_squares.
    Per-column scale/offset parameters are solved analytically at each residual
    evaluation.

    Current prototype limitations:
    - fixed initial conditions only
    - no tied parameters
    - one observed species shared by all columns
    - unconstrained linear scale/offset projection
    """
    engine_bundle = resolve_engine_bundle(
        engine_name=engine_name,
        engine_bundle=engine_bundle,
    )

    if settings is None:
        settings = FitSettings(
            species_mapping={},
        )

    if signal_columns is None:
        signal_columns = dataset.signal_columns

    if observed_species not in model.species:
        raise ValueError(
            f"Observed species is not present in model: {observed_species}"
        )

    timepoints = dataset.raw_dataframe[dataset.time_column].to_numpy(dtype=float)
    observed_dataframe = dataset.raw_dataframe

    initial_parameters = _build_initial_parameter_dict(parameter_specs)
    initial_conditions = _build_fixed_initial_conditions(initial_condition_specs)

    free_specs = _free_parameter_specs(parameter_specs)

    x0 = np.asarray(
        [spec.initial_guess for spec in free_specs],
        dtype=float,
    )

    lower_bounds = np.asarray(
        [spec.lower_bound for spec in free_specs],
        dtype=float,
    )

    upper_bounds = np.asarray(
        [spec.upper_bound for spec in free_specs],
        dtype=float,
    )

    last_simulation_result: ArraySolveResult | None = None
    last_projection_result: LinearObservableProjectionResult | None = None

    def residual_function(free_values: np.ndarray) -> np.ndarray:
        nonlocal last_simulation_result
        nonlocal last_projection_result

        parameter_values = _parameter_vector_to_dict(
            parameter_specs=parameter_specs,
            free_values=free_values,
        )

        simulation_result = solve_array_mass_action_model(
            model=model,
            parameters=parameter_values,
            initial_conditions=initial_conditions,
            timepoints=timepoints,
            backend=backend,
            method=method,
            rtol=settings.rtol,
            atol=settings.atol,
        )

        if not simulation_result.success:
            return np.full(
                len(timepoints) * len(signal_columns),
                1e12,
                dtype=float,
            )

        species_values = simulation_result.get_species_values(observed_species)

        projection_result = project_observables_onto_species(
            timepoints=timepoints,
            simulated_species_values=species_values,
            observed_dataframe=observed_dataframe,
            signal_columns=signal_columns,
            fit_scale=fit_scale,
            fit_offset=fit_offset,
            signal_weights=settings.signal_weights,
            engine_bundle=engine_bundle,
        )

        last_simulation_result = simulation_result
        last_projection_result = projection_result

        return projection_result.residual_vector

    optimizer_result = engine_least_squares(
        engine_bundle=engine_bundle,
        residual_function=residual_function,
        x0=x0,
        bounds=(lower_bounds, upper_bounds),
        method=settings.method,
        loss=settings.loss,
        max_nfev=settings.max_nfev,
    )

    fitted_parameters = _parameter_vector_to_dict(
        parameter_specs=parameter_specs,
        free_values=optimizer_result.x,
    )

    final_simulation_result = solve_array_mass_action_model(
        model=model,
        parameters=fitted_parameters,
        initial_conditions=initial_conditions,
        timepoints=timepoints,
        backend=backend,
        method=method,
        rtol=settings.rtol,
        atol=settings.atol,
    )

    final_species_values = final_simulation_result.get_species_values(observed_species)

    final_projection_result = project_observables_onto_species(
        timepoints=timepoints,
        simulated_species_values=final_species_values,
        observed_dataframe=observed_dataframe,
        signal_columns=signal_columns,
        fit_scale=fit_scale,
        fit_offset=fit_offset,
        signal_weights=settings.signal_weights,
        engine_bundle=engine_bundle,
    )

    statistics = _build_fit_statistics(
        rss=final_projection_result.rss,
        n_observations=final_projection_result.n_observations,
        n_nonlinear_parameters=len(free_specs),
        n_linear_parameters=final_projection_result.n_linear_parameters,
    )

    simulation_dataframe = _simulation_to_dataframe(final_simulation_result)

    return VariableProjectionGlobalObservableFitResult(
        success=bool(optimizer_result.success and final_simulation_result.success),
        message=str(optimizer_result.message),
        fitted_parameters=fitted_parameters,
        initial_parameters=initial_parameters,
        fitted_initial_conditions=initial_conditions,
        statistics=statistics,
        observable_table=final_projection_result.observable_table,
        predicted_dataframe=final_projection_result.predicted_dataframe,
        residuals_dataframe=final_projection_result.residuals_dataframe,
        simulation_dataframe=simulation_dataframe,
        residual_vector=final_projection_result.residual_vector,
        simulation_result=final_simulation_result,
        cost=float(optimizer_result.cost),
        nfev=int(optimizer_result.nfev),
        status=int(optimizer_result.status),
        optimality=float(optimizer_result.optimality),
        active_mask=optimizer_result.active_mask,
    )


def export_variable_projection_fit(
    result: VariableProjectionGlobalObservableFitResult,
    output_dir: str,
) -> dict[str, str]:
    """
    Export variable-projection fit outputs.
    """

    from pathlib import Path

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    written_files: dict[str, str] = {}

    observable_path = output_path / "projected_observables.csv"
    predicted_path = output_path / "projected_predictions.csv"
    residuals_path = output_path / "projected_residuals.csv"
    simulation_path = output_path / "projected_simulation.csv"
    statistics_path = output_path / "projected_fit_statistics.csv"
    parameters_path = output_path / "projected_fitted_parameters.csv"

    result.observable_table.to_csv(observable_path, index=False)
    result.predicted_dataframe.to_csv(predicted_path, index=False)
    result.residuals_dataframe.to_csv(residuals_path, index=False)
    result.simulation_dataframe.to_csv(simulation_path, index=False)

    pd.DataFrame([result.statistics]).to_csv(statistics_path, index=False)

    pd.DataFrame(
        [
            {
                "parameter": name,
                "value": value,
            }
            for name, value in result.fitted_parameters.items()
        ]
    ).to_csv(parameters_path, index=False)

    written_files["projected_observables"] = str(observable_path)
    written_files["projected_predictions"] = str(predicted_path)
    written_files["projected_residuals"] = str(residuals_path)
    written_files["projected_simulation"] = str(simulation_path)
    written_files["projected_fit_statistics"] = str(statistics_path)
    written_files["projected_fitted_parameters"] = str(parameters_path)

    return written_files
