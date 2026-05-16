from scipy.optimize import least_squares

from odefit.data.dataset import Dataset
from odefit.fitting.fit_result import FitResult
from odefit.fitting.fit_settings import FitSettings
from odefit.fitting.initial_condition_spec import InitialConditionSpec
from odefit.fitting.initial_condition_vector import build_initial_condition_dict
from odefit.fitting.objective import objective_function
from odefit.fitting.optimization_vector import (
    build_optimization_bounds,
    build_optimization_vector,
    vector_to_model_inputs,
)
from odefit.fitting.parameter_spec import ParameterSpec
from odefit.fitting.parameter_table import build_initial_parameter_dict
from odefit.fitting.parameter_vector import get_free_parameter_specs
from odefit.fitting.statistics import calculate_fit_statistics
from odefit.fitting.validation import validate_fit_inputs
from odefit.model.model_spec import ModelSpec
from odefit.simulation.solver import simulate_model


def fit_model(
    model: ModelSpec,
    dataset: Dataset,
    parameter_specs: list[ParameterSpec],
    initial_conditions: dict[str, float] | None = None,
    settings: FitSettings | None = None,
    initial_condition_specs: list[InitialConditionSpec] | None = None,
) -> FitResult:
    """
    Fit a model to a dataset.

    Backward-compatible usage:
        pass initial_conditions as a dict

    New usage:
        pass initial_condition_specs to allow fitted initial conditions
    """

    resolved_initial_condition_specs = validate_fit_inputs(
        model=model,
        dataset=dataset,
        parameter_specs=parameter_specs,
        initial_conditions=initial_conditions,
        settings=settings,
        initial_condition_specs=initial_condition_specs,
    )

    if settings is None:
        raise ValueError("FitSettings are required")

    initial_vector = build_optimization_vector(
        parameter_specs=parameter_specs,
        initial_condition_specs=resolved_initial_condition_specs,
    )

    bounds = build_optimization_bounds(
        parameter_specs=parameter_specs,
        initial_condition_specs=resolved_initial_condition_specs,
    )

    result = least_squares(
        fun=objective_function,
        x0=initial_vector,
        bounds=bounds,
        args=(
            model,
            dataset,
            parameter_specs,
            resolved_initial_condition_specs,
            settings,
        ),
        method=settings.method,
        loss=settings.loss,
        max_nfev=settings.max_nfev,
    )

    fitted_parameters, fitted_initial_conditions = vector_to_model_inputs(
        vector=result.x,
        parameter_specs=parameter_specs,
        initial_condition_specs=resolved_initial_condition_specs,
    )

    initial_parameters = build_initial_parameter_dict(parameter_specs)

    initial_condition_values = build_initial_condition_dict(
        resolved_initial_condition_specs
    )

    simulation_result = simulate_model(
        model=model,
        parameters=fitted_parameters,
        initial_conditions=fitted_initial_conditions,
        timepoints=dataset.time_values,
    )

    residuals = objective_function(
        optimization_vector=result.x,
        model=model,
        dataset=dataset,
        parameter_specs=parameter_specs,
        initial_condition_specs=resolved_initial_condition_specs,
        settings=settings,
    )

    number_of_free_parameters = len(get_free_parameter_specs(parameter_specs))

    statistics = calculate_fit_statistics(
        residuals=residuals,
        number_of_parameters=number_of_free_parameters,
    )

    return FitResult(
        success=result.success,
        message=result.message,
        fitted_parameters=fitted_parameters,
        initial_parameters=initial_parameters,
        residuals=residuals,
        statistics=statistics,
        simulation_result=simulation_result,
        nfev=result.nfev,
        cost=float(result.cost),
        fitted_initial_conditions=fitted_initial_conditions,
        initial_conditions=initial_condition_values,
    )
