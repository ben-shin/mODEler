from scipy.optimize import least_squares

from odefit.data.dataset import Dataset
from odefit.fitting.fit_result import FitResult
from odefit.fitting.fit_settings import FitSettings
from odefit.fitting.objective import objective_function
from odefit.fitting.parameter_spec import ParameterSpec
from odefit.fitting.parameter_table import build_initial_parameter_dict
from odefit.fitting.parameter_vector import (
    build_bounds,
    build_initial_vector,
    get_free_parameter_specs,
    vector_to_parameter_dict,
)
from odefit.fitting.statistics import calculate_fit_statistics
from odefit.fitting.validation import validate_fit_inputs
from odefit.model.model_spec import ModelSpec
from odefit.simulation.solver import simulate_model


def fit_model(
    model: ModelSpec,
    dataset: Dataset,
    parameter_specs: list[ParameterSpec],
    initial_conditions: dict[str, float],
    settings: FitSettings,
) -> FitResult:
    """
    Fit a model to a dataset.
    """

    validate_fit_inputs(
        model=model,
        dataset=dataset,
        parameter_specs=parameter_specs,
        initial_conditions=initial_conditions,
        settings=settings,
    )

    initial_vector = build_initial_vector(parameter_specs)
    bounds = build_bounds(parameter_specs)

    result = least_squares(
        fun=objective_function,
        x0=initial_vector,
        bounds=bounds,
        args=(
            model,
            dataset,
            parameter_specs,
            initial_conditions,
            settings,
        ),
        method=settings.method,
        loss=settings.loss,
        max_nfev=settings.max_nfev,
    )

    fitted_parameters = vector_to_parameter_dict(
        vector=result.x,
        parameter_specs=parameter_specs,
    )

    initial_parameters = build_initial_parameter_dict(parameter_specs)

    simulation_result = simulate_model(
        model=model,
        parameters=fitted_parameters,
        initial_conditions=initial_conditions,
        timepoints=dataset.time_values,
    )

    residuals = objective_function(
        parameter_vector=result.x,
        model=model,
        dataset=dataset,
        parameter_specs=parameter_specs,
        initial_conditions=initial_conditions,
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
    )
