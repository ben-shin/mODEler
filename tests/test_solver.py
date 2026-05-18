import numpy as np
import pytest

from odefit.model.model_spec import build_model_spec
from odefit.simulation.simulation_settings import SimulationSettings
from odefit.simulation.solver import (
    build_initial_value_vector,
    detect_negative_value_warnings,
    simulate_model,
    validate_solver_method,
    validate_timepoints,
)


def test_validate_timepoints_accepts_strictly_increasing_values():
    validate_timepoints(np.array([0.0, 1.0, 2.0]))


def test_validate_timepoints_rejects_single_timepoint():
    with pytest.raises(ValueError):
        validate_timepoints(np.array([0.0]))


def test_validate_timepoints_rejects_repeated_timepoints():
    with pytest.raises(ValueError):
        validate_timepoints(np.array([0.0, 1.0, 1.0]))


def test_validate_timepoints_rejects_decreasing_timepoints():
    with pytest.raises(ValueError):
        validate_timepoints(np.array([0.0, 2.0, 1.0]))


def test_validate_solver_method_accepts_supported_methods():
    for method in ["RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"]:
        validate_solver_method(method)


def test_validate_solver_method_rejects_unknown_method():
    with pytest.raises(ValueError):
        validate_solver_method("not_a_solver")


def test_build_initial_value_vector_uses_model_species_order():
    model = build_model_spec(
        """
B>C
A>B
"""
    )

    vector = build_initial_value_vector(
        model=model,
        initial_conditions={
            "A": 1.0,
            "B": 2.0,
            "C": 3.0,
        },
    )

    assert np.allclose(vector, [1.0, 2.0, 3.0])


def test_simulate_irreversible_decay():
    model = build_model_spec("A>B")

    result = simulate_model(
        model=model,
        parameters={
            "k1f": 1.0,
        },
        initial_conditions={
            "A": 1.0,
            "B": 0.0,
        },
        timepoints=np.array([0.0, 1.0, 2.0]),
    )

    assert result.success
    assert result.values.shape == (3, 2)

    a_values = result.get_species_values("A")
    b_values = result.get_species_values("B")

    assert a_values[0] == pytest.approx(1.0)
    assert b_values[0] == pytest.approx(0.0)

    assert a_values[-1] < a_values[0]
    assert b_values[-1] > b_values[0]


def test_simulate_reversible_mass_conservation():
    model = build_model_spec("A-B")

    result = simulate_model(
        model=model,
        parameters={
            "k1f": 1.0,
            "k1r": 0.5,
        },
        initial_conditions={
            "A": 1.0,
            "B": 0.0,
        },
        timepoints=np.linspace(0.0, 10.0, 20),
    )

    total = result.get_species_values("A") + result.get_species_values("B")

    assert np.allclose(total, 1.0, atol=1e-6)


def test_simulate_missing_parameter_raises_error():
    model = build_model_spec("A>B")

    with pytest.raises(ValueError):
        simulate_model(
            model=model,
            parameters={},
            initial_conditions={
                "A": 1.0,
                "B": 0.0,
            },
            timepoints=np.array([0.0, 1.0]),
        )


def test_simulate_missing_initial_condition_raises_error():
    model = build_model_spec("A>B")

    with pytest.raises(ValueError):
        simulate_model(
            model=model,
            parameters={
                "k1f": 1.0,
            },
            initial_conditions={
                "A": 1.0,
            },
            timepoints=np.array([0.0, 1.0]),
        )


def test_simulate_rejects_non_increasing_timepoints():
    model = build_model_spec("A>B")

    with pytest.raises(ValueError):
        simulate_model(
            model=model,
            parameters={
                "k1f": 1.0,
            },
            initial_conditions={
                "A": 1.0,
                "B": 0.0,
            },
            timepoints=np.array([0.0, 1.0, 1.0]),
        )


def test_simulate_with_solver_method_selection():
    model = build_model_spec("A>B")

    for method in ["RK45", "BDF", "Radau", "LSODA"]:
        result = simulate_model(
            model=model,
            parameters={
                "k1f": 1.0,
            },
            initial_conditions={
                "A": 1.0,
                "B": 0.0,
            },
            timepoints=np.linspace(0.0, 2.0, 5),
            settings=SimulationSettings(
                method=method,
                rtol=1e-7,
                atol=1e-9,
            ),
        )

        assert result.success
        assert result.values.shape == (5, 2)


def test_stiff_solver_bdf_handles_fast_reaction():
    model = build_model_spec("A>B")

    result = simulate_model(
        model=model,
        parameters={
            "k1f": 1000.0,
        },
        initial_conditions={
            "A": 1.0,
            "B": 0.0,
        },
        timepoints=np.linspace(0.0, 0.1, 10),
        settings=SimulationSettings(
            method="BDF",
            rtol=1e-6,
            atol=1e-9,
        ),
    )

    assert result.success

    a_values = result.get_species_values("A")
    b_values = result.get_species_values("B")

    assert a_values[-1] < 1e-3
    assert b_values[-1] > 0.999


def test_stiff_solver_radau_handles_fast_reaction():
    model = build_model_spec("A>B")

    result = simulate_model(
        model=model,
        parameters={
            "k1f": 1000.0,
        },
        initial_conditions={
            "A": 1.0,
            "B": 0.0,
        },
        timepoints=np.linspace(0.0, 0.1, 10),
        settings=SimulationSettings(
            method="Radau",
            rtol=1e-6,
            atol=1e-9,
        ),
    )

    assert result.success

    a_values = result.get_species_values("A")

    assert a_values[-1] < 1e-3


def test_detect_negative_value_warnings():
    values = np.array(
        [
            [1.0, 0.0],
            [0.5, -1e-3],
        ]
    )

    warnings = detect_negative_value_warnings(
        values=values,
        species=["A", "B"],
        tolerance=1e-9,
    )

    assert len(warnings) == 1
    assert "Species B" in warnings[0]
    assert "negative simulated value" in warnings[0]


def test_detect_negative_value_warnings_ignores_tiny_values_within_tolerance():
    values = np.array(
        [
            [1.0, 0.0],
            [0.5, -1e-12],
        ]
    )

    warnings = detect_negative_value_warnings(
        values=values,
        species=["A", "B"],
        tolerance=1e-9,
    )

    assert warnings == []


def test_simulation_result_contains_solver_metadata():
    model = build_model_spec("A>B")

    result = simulate_model(
        model=model,
        parameters={
            "k1f": 1.0,
        },
        initial_conditions={
            "A": 1.0,
            "B": 0.0,
        },
        timepoints=np.array([0.0, 1.0]),
    )

    assert isinstance(result.success, bool)
    assert isinstance(result.message, str)
    assert isinstance(result.warnings, list)


def test_clip_negative_concentrations_setting_does_not_crash():
    model = build_model_spec("A>B")

    result = simulate_model(
        model=model,
        parameters={
            "k1f": 1.0,
        },
        initial_conditions={
            "A": 1.0,
            "B": 0.0,
        },
        timepoints=np.array([0.0, 1.0, 2.0]),
        settings=SimulationSettings(
            clip_negative_concentrations=True,
        ),
    )

    assert result.success
