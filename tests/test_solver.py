import numpy as np
import pytest

from odefit.model.model_spec import build_model_spec
from odefit.simulation.solver import simulate_model


def test_simulate_irreversible_decay():
    model = build_model_spec("A>B")

    parameters = {
        "k1f": 0.5,
    }

    initial_conditions = {
        "A": 1.0,
        "B": 0.0,
    }

    timepoints = np.array([0.0, 1.0, 2.0])

    result = simulate_model(
        model=model,
        parameters=parameters,
        initial_conditions=initial_conditions,
        timepoints=timepoints,
    )

    a_values = result.get_species_values("A")
    b_values = result.get_species_values("B")

    expected_a = np.exp(-0.5 * timepoints)

    assert np.allclose(a_values, expected_a, atol=1e-4)
    assert np.allclose(a_values + b_values, 1.0, atol=1e-6)


def test_simulate_reversible_reaction_conserves_total_mass():
    model = build_model_spec("A-B")

    parameters = {
        "k1f": 0.5,
        "k1r": 0.2,
    }

    initial_conditions = {
        "A": 1.0,
        "B": 0.0,
    }

    timepoints = np.linspace(0, 10, 20)

    result = simulate_model(
        model=model,
        parameters=parameters,
        initial_conditions=initial_conditions,
        timepoints=timepoints,
    )

    a_values = result.get_species_values("A")
    b_values = result.get_species_values("B")

    assert np.allclose(a_values + b_values, 1.0, atol=1e-6)


def test_missing_parameter_raises_error():
    model = build_model_spec("A-B")

    parameters = {
        "k1f": 0.5,
    }

    initial_conditions = {
        "A": 1.0,
        "B": 0.0,
    }

    timepoints = [0.0, 1.0]

    with pytest.raises(ValueError):
        simulate_model(
            model=model,
            parameters=parameters,
            initial_conditions=initial_conditions,
            timepoints=timepoints,
        )


def test_missing_initial_condition_raises_error():
    model = build_model_spec("A>B")

    parameters = {
        "k1f": 0.5,
    }

    initial_conditions = {
        "A": 1.0,
    }

    timepoints = [0.0, 1.0]

    with pytest.raises(ValueError):
        simulate_model(
            model=model,
            parameters=parameters,
            initial_conditions=initial_conditions,
            timepoints=timepoints,
        )


def test_timepoints_must_be_increasing():
    model = build_model_spec("A>B")

    parameters = {
        "k1f": 0.5,
    }

    initial_conditions = {
        "A": 1.0,
        "B": 0.0,
    }

    timepoints = [0.0, 2.0, 1.0]

    with pytest.raises(ValueError):
        simulate_model(
            model=model,
            parameters=parameters,
            initial_conditions=initial_conditions,
            timepoints=timepoints,
        )
