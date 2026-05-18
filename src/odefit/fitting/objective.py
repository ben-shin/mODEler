import numpy as np

from odefit.data.dataset import Dataset
from odefit.fitting.fit_settings import FitSettings
from odefit.fitting.initial_condition_spec import InitialConditionSpec
from odefit.fitting.observable_residuals import calculate_observable_residuals
from odefit.fitting.observable_spec import ObservableSpec
from odefit.fitting.optimization_vector import (
    vector_to_fit_inputs,
    vector_to_model_inputs,
)
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
    observable_specs: list[ObservableSpec] | None = None,
) -> np.ndarray:
    """
    Objective function for least-squares fitting.

    If observable_specs are provided, residuals are calculated with:

        observed = scale * species + offset

    Otherwise, old direct species_mapping behavior is used.
    """

    if observable_specs is None:
        parameters, initial_conditions = vector_to_model_inputs(
            vector=optimization_vector,
            parameter_specs=parameter_specs,
            initial_condition_specs=initial_condition_specs,
        )

        observable_parameters = None

    else:
        parameters, initial_conditions, observable_parameters = vector_to_fit_inputs(
            vector=optimization_vector,
            parameter_specs=parameter_specs,
            initial_condition_specs=initial_condition_specs,
            observable_specs=observable_specs,
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

    if observable_parameters is not None:
        return calculate_observable_residuals(
            dataset=dataset,
            simulation_result=simulation_result,
            observable_parameters=observable_parameters,
            use_normalized_data=settings.use_normalized_data,
            signal_weights=settings.signal_weights,
        )

    return calculate_residuals(
        dataset=dataset,
        simulation_result=simulation_result,
        species_mapping=settings.species_mapping,
        use_normalized_data=settings.use_normalized_data,
        signal_weights=settings.signal_weights,
    )
