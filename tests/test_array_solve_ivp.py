import numpy as np
import pytest

from odefit.model.model_spec import build_model_spec
from odefit.performance.array_solve_ivp import (
    make_array_rhs_function,
    solve_array_mass_action_model,
)
from odefit.performance.array_rhs import (
    compile_mass_action_model,
    parameter_dict_to_array,
)
from odefit.performance.numba_rhs import is_numba_available


def test_solve_array_mass_action_first_order_numpy():
    model = build_model_spec("A>B")

    timepoints = np.linspace(0.0, 5.0, 51)
    k = 0.4

    result = solve_array_mass_action_model(
        model=model,
        parameters={
            "k1f": k,
        },
        initial_conditions={
            "A": 1.0,
            "B": 0.0,
        },
        timepoints=timepoints,
        backend="numpy",
        method="LSODA",
        rtol=1e-9,
        atol=1e-11,
    )

    expected_a = np.exp(-k * timepoints)
    expected_b = 1.0 - expected_a

    assert result.success
    assert result.backend == "numpy"
    assert result.species == ["A", "B"]

    np.testing.assert_allclose(
        result.get_species_values("A"),
        expected_a,
        rtol=1e-5,
        atol=1e-7,
    )

    np.testing.assert_allclose(
        result.get_species_values("B"),
        expected_b,
        rtol=1e-5,
        atol=1e-7,
    )


def test_solve_array_mass_action_reversible_conserves_mass_numpy():
    model = build_model_spec("A-B")

    timepoints = np.linspace(0.0, 5.0, 51)

    result = solve_array_mass_action_model(
        model=model,
        parameters={
            "k1f": 0.4,
            "k1r": 0.2,
        },
        initial_conditions={
            "A": 1.0,
            "B": 0.0,
        },
        timepoints=timepoints,
        backend="numpy",
        method="LSODA",
    )

    total = result.get_species_values("A") + result.get_species_values("B")

    assert result.success
    np.testing.assert_allclose(
        total,
        np.ones_like(total),
        rtol=1e-6,
        atol=1e-8,
    )


def test_make_array_rhs_function_rejects_unknown_backend():
    model = build_model_spec("A>B")
    compiled = compile_mass_action_model(model)

    parameter_array = parameter_dict_to_array(
        compiled,
        {
            "k1f": 0.1,
        },
    )

    with pytest.raises(ValueError, match="Unknown array RHS backend"):
        make_array_rhs_function(
            compiled_model=compiled,
            parameter_array=parameter_array,
            backend="not_a_backend",
        )


def test_solve_array_mass_action_rejects_bad_timepoints():
    model = build_model_spec("A>B")

    with pytest.raises(ValueError, match="strictly increasing"):
        solve_array_mass_action_model(
            model=model,
            parameters={
                "k1f": 0.1,
            },
            initial_conditions={
                "A": 1.0,
                "B": 0.0,
            },
            timepoints=np.array([0.0, 1.0, 1.0]),
            backend="numpy",
        )


def test_solve_array_mass_action_rejects_missing_parameter():
    model = build_model_spec("A>B")

    with pytest.raises(ValueError, match="Missing parameter values"):
        solve_array_mass_action_model(
            model=model,
            parameters={},
            initial_conditions={
                "A": 1.0,
                "B": 0.0,
            },
            timepoints=np.array([0.0, 1.0]),
            backend="numpy",
        )


def test_solve_array_mass_action_get_species_values_rejects_missing_species():
    model = build_model_spec("A>B")

    result = solve_array_mass_action_model(
        model=model,
        parameters={
            "k1f": 0.1,
        },
        initial_conditions={
            "A": 1.0,
            "B": 0.0,
        },
        timepoints=np.array([0.0, 1.0]),
        backend="numpy",
    )

    with pytest.raises(ValueError, match="Species not found"):
        result.get_species_values("missing")


@pytest.mark.skipif(
    not is_numba_available(),
    reason="numba is not installed",
)
def test_solve_array_mass_action_first_order_numba_matches_numpy():
    model = build_model_spec("A>B")

    timepoints = np.linspace(0.0, 5.0, 51)

    numpy_result = solve_array_mass_action_model(
        model=model,
        parameters={
            "k1f": 0.4,
        },
        initial_conditions={
            "A": 1.0,
            "B": 0.0,
        },
        timepoints=timepoints,
        backend="numpy",
        method="LSODA",
        rtol=1e-9,
        atol=1e-11,
    )

    numba_result = solve_array_mass_action_model(
        model=model,
        parameters={
            "k1f": 0.4,
        },
        initial_conditions={
            "A": 1.0,
            "B": 0.0,
        },
        timepoints=timepoints,
        backend="numba",
        method="LSODA",
        rtol=1e-9,
        atol=1e-11,
    )

    assert numba_result.success

    np.testing.assert_allclose(
        numba_result.values,
        numpy_result.values,
        rtol=1e-6,
        atol=1e-8,
    )
