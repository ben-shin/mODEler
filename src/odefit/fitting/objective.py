import numpy as np

from odefit.data.dataset import Dataset
from odefit.fitting.fit_settings import FitSettings
from odefit.fitting.initial_condition_spec import InitialConditionSpec
from odefit.fitting.optimization_vector import vector_to_model_inputs
from odefit.fitting.parameter_spec import ParameterSpec
from odefit.fitting.residuals import calculate_residuals
from odefit.model.model_spec import ModelSpec
from odefit.simulation.simulation_settings import SimulationSettings
from odefit.simulation.solver import simulate_model


def objective_function(
    optimization_vector: np.ndarray,
    model: ModelSpec,
    dataset: Dataset,
    parameter_specs: list[ParameterSpec],
    initial_condition_specs: list[InitialConditionSpec],
    settings: FitSettings,
) -> np.ndarray:
    """
    Objective function for least-squares fitting.

    The optimization vector contains both:
    - free kinetic parameters
    - free initial conditions
    """

    parameters, initial_conditions = vector_to_model_inputs(
        vector=optimization_vector,
        parameter_specs=parameter_specs,
        initial_condition_specs=initial_condition_specs,
    )

    simulation_settings = SimulationSettings(
        rtol=settings.rtol,
        atol=settings.atol,
    )

    simulation_result = simulate_model(
        model=model,
        parameters=parameters,
        initial_conditions=initial_conditions,
        timepoints=dataset.time_values,
        settings=simulation_settings,
    )

    return calculate_residuals(
        dataset=dataset,
        simulation_result=simulation_result,
        species_mapping=settings.species_mapping,
        use_normalized_data=settings.use_normalized_data,
    )
