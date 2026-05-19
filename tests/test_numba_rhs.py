import numpy as np
import pytest

from odefit.model.model_spec import build_model_spec
from odefit.performance.array_rhs import (
    compile_mass_action_model,
    concentration_dict_to_array,
    evaluate_mass_action_rates,
    evaluate_mass_action_rhs,
    parameter_dict_to_array,
)
from odefit.performance.numba_rhs import (
    evaluate_mass_action_rates_numba,
    evaluate_mass_action_rhs_numba,
    is_numba_available,
    warm_up_numba_rhs,
)


def test_is_numba_available_returns_bool():
    assert isinstance(is_numba_available(), bool)


numba = pytest.importorskip("numba")


def test_numba_irreversible_rhs_matches_numpy_rhs():
    model = build_model_spec("A>B")
    compiled = compile_mass_action_model(model)

    y = np.array([2.0, 0.0])
    p = np.array([0.5])

    expected_rates = evaluate_mass_action_rates(
        compiled_model=compiled,
        concentrations=y,
        parameters=p,
    )

    expected_rhs = evaluate_mass_action_rhs(
        compiled_model=compiled,
        concentrations=y,
        parameters=p,
    )

    warm_up_numba_rhs(compiled, y, p)

    observed_rates = evaluate_mass_action_rates_numba(
        compiled_model=compiled,
        concentrations=y,
        parameters=p,
    )

    observed_rhs = evaluate_mass_action_rhs_numba(
        compiled_model=compiled,
        concentrations=y,
        parameters=p,
    )

    np.testing.assert_allclose(observed_rates, expected_rates)
    np.testing.assert_allclose(observed_rhs, expected_rhs)


def test_numba_reversible_rhs_matches_numpy_rhs():
    model = build_model_spec("A-B")
    compiled = compile_mass_action_model(model)

    y = np.array([2.0, 3.0])
    p = np.array([0.5, 0.25])

    expected_rhs = evaluate_mass_action_rhs(
        compiled_model=compiled,
        concentrations=y,
        parameters=p,
    )

    warm_up_numba_rhs(compiled, y, p)

    observed_rhs = evaluate_mass_action_rhs_numba(
        compiled_model=compiled,
        concentrations=y,
        parameters=p,
    )

    np.testing.assert_allclose(observed_rhs, expected_rhs)


def test_numba_second_order_rhs_matches_numpy_rhs():
    model = build_model_spec("2A>A2")
    compiled = compile_mass_action_model(model)

    y = concentration_dict_to_array(
        compiled,
        {
            "A": 3.0,
            "A2": 0.0,
        },
    )

    p = parameter_dict_to_array(
        compiled,
        {
            "k1f": 0.2,
        },
    )

    expected_rhs = evaluate_mass_action_rhs(
        compiled_model=compiled,
        concentrations=y,
        parameters=p,
    )

    warm_up_numba_rhs(compiled, y, p)

    observed_rhs = evaluate_mass_action_rhs_numba(
        compiled_model=compiled,
        concentrations=y,
        parameters=p,
    )

    np.testing.assert_allclose(observed_rhs, expected_rhs)


def test_numba_bimolecular_rhs_matches_numpy_rhs():
    model = build_model_spec("A+B>C")
    compiled = compile_mass_action_model(model)

    y = concentration_dict_to_array(
        compiled,
        {
            "A": 2.0,
            "B": 3.0,
            "C": 0.0,
        },
    )

    p = parameter_dict_to_array(
        compiled,
        {
            "k1f": 0.5,
        },
    )

    expected_rhs = evaluate_mass_action_rhs(
        compiled_model=compiled,
        concentrations=y,
        parameters=p,
    )

    warm_up_numba_rhs(compiled, y, p)

    observed_rhs = evaluate_mass_action_rhs_numba(
        compiled_model=compiled,
        concentrations=y,
        parameters=p,
    )

    np.testing.assert_allclose(observed_rhs, expected_rhs)


def test_numba_clip_negative_concentrations_matches_numpy_rhs():
    model = build_model_spec("A>B")
    compiled = compile_mass_action_model(model)

    y = np.array([-2.0, 0.0])
    p = np.array([0.5])

    expected_rhs = evaluate_mass_action_rhs(
        compiled_model=compiled,
        concentrations=y,
        parameters=p,
        clip_negative_concentrations=True,
    )

    warm_up_numba_rhs(compiled, np.array([0.0, 0.0]), p)

    observed_rhs = evaluate_mass_action_rhs_numba(
        compiled_model=compiled,
        concentrations=y,
        parameters=p,
        clip_negative_concentrations=True,
    )

    np.testing.assert_allclose(observed_rhs, expected_rhs)


def test_numba_rejects_wrong_concentration_shape():
    model = build_model_spec("A>B")
    compiled = compile_mass_action_model(model)

    with pytest.raises(ValueError, match="concentrations must have shape"):
        evaluate_mass_action_rhs_numba(
            compiled_model=compiled,
            concentrations=np.array([1.0]),
            parameters=np.array([0.1]),
        )


def test_numba_rejects_wrong_parameter_shape():
    model = build_model_spec("A>B")
    compiled = compile_mass_action_model(model)

    with pytest.raises(ValueError, match="parameters must have shape"):
        evaluate_mass_action_rhs_numba(
            compiled_model=compiled,
            concentrations=np.array([1.0, 0.0]),
            parameters=np.array([0.1, 0.2]),
        )
