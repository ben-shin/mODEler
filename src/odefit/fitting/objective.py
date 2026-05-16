import numpy as np

from odefit.data.dataset import Dataset
from odefit.fitting.fit_settings import FitSettings
from odefit.fitting.parameter_spec import ParameterSpec
from odefit.fitting.parameter_vector import vector_to_parameter_dict
from odefit.fitting.residuals import calculate_residuals
from odefit.model.model_spec import ModelSpec
from odefit.simulation.simulation_settings import SimulationSettings
from odefit.simulation.solver import simulate_model


def objective_function(
    parameter_vector: np.ndarray,
    model: ModelSpec,
    dataset: Dataset,
    parameter_specs: list[ParameterSpec],
    initial_conditions: dict[str, float],
    settings: FitSettings,
) -> np.ndarray:
    """
    Objective function for least-squares fitting.
    """

    parameters = vector_to_parameter_dict(
        vector=parameter_vector,
        parameter_specs=parameter_specs,
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
