import numpy as np
import pytest

from odefit.model.model_spec import build_model_spec
from odefit.performance.array_rhs import (
    array_to_species_dict,
    compile_mass_action_model,
    concentration_dict_to_array,
    evaluate_mass_action_rates,
    evaluate_mass_action_rhs,
    parameter_dict_to_array,
)


def test_compile_irreversible_first_order_model():
    model = build_model_spec("A>B")

    compiled = compile_mass_action_model(model)

    assert compiled.species == ["A", "B"]
    assert compiled.parameters == ["k1f"]
    assert compiled.n_processes == 1

    np.testing.assert_allclose(
        compiled.reactant_orders,
        np.array([[1.0, 0.0]]),
    )

    np.testing.assert_allclose(
        compiled.net_stoichiometry,
        np.array([[-1.0, 1.0]]),
    )

    np.testing.assert_array_equal(
        compiled.rate_parameter_indices,
        np.array([0]),
    )


def test_evaluate_irreversible_first_order_rhs():
    model = build_model_spec("A>B")
    compiled = compile_mass_action_model(model)

    y = np.array([2.0, 0.0])
    p = np.array([0.5])

    rates = evaluate_mass_action_rates(
        compiled_model=compiled,
        concentrations=y,
        parameters=p,
    )

    rhs = evaluate_mass_action_rhs(
        compiled_model=compiled,
        concentrations=y,
        parameters=p,
    )

    np.testing.assert_allclose(rates, np.array([1.0]))
    np.testing.assert_allclose(rhs, np.array([-1.0, 1.0]))


def test_compile_reversible_first_order_model():
    model = build_model_spec("A-B")

    compiled = compile_mass_action_model(model)

    assert compiled.species == ["A", "B"]
    assert compiled.parameters == ["k1f", "k1r"]
    assert compiled.n_processes == 2

    np.testing.assert_allclose(
        compiled.reactant_orders,
        np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        ),
    )

    np.testing.assert_allclose(
        compiled.net_stoichiometry,
        np.array(
            [
                [-1.0, 1.0],
                [1.0, -1.0],
            ]
        ),
    )

    np.testing.assert_array_equal(
        compiled.rate_parameter_indices,
        np.array([0, 1]),
    )


def test_evaluate_reversible_first_order_rhs():
    model = build_model_spec("A-B")
    compiled = compile_mass_action_model(model)

    y = np.array([2.0, 3.0])
    p = np.array([0.5, 0.25])

    rates = evaluate_mass_action_rates(
        compiled_model=compiled,
        concentrations=y,
        parameters=p,
    )

    rhs = evaluate_mass_action_rhs(
        compiled_model=compiled,
        concentrations=y,
        parameters=p,
    )

    # forward = 0.5 * A = 1.0
    # reverse = 0.25 * B = 0.75
    # net dA = -1.0 + 0.75 = -0.25
    # net dB = 1.0 - 0.75 = 0.25
    np.testing.assert_allclose(rates, np.array([1.0, 0.75]))
    np.testing.assert_allclose(rhs, np.array([-0.25, 0.25]))


def test_evaluate_second_order_rhs():
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

    rates = evaluate_mass_action_rates(
        compiled_model=compiled,
        concentrations=y,
        parameters=p,
    )

    rhs = evaluate_mass_action_rhs(
        compiled_model=compiled,
        concentrations=y,
        parameters=p,
    )

    # rate = 0.2 * A^2 = 1.8
    # dA = -2 * 1.8 = -3.6
    # dA2 = +1 * 1.8 = 1.8
    np.testing.assert_allclose(rates, np.array([1.8]))
    rhs_dict = array_to_species_dict(compiled, rhs)

    assert rhs_dict["A"] == pytest.approx(-3.6)
    assert rhs_dict["A2"] == pytest.approx(1.8)


def test_evaluate_bimolecular_rhs():
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

    rates = evaluate_mass_action_rates(
        compiled_model=compiled,
        concentrations=y,
        parameters=p,
    )

    rhs = evaluate_mass_action_rhs(
        compiled_model=compiled,
        concentrations=y,
        parameters=p,
    )

    # rate = 0.5 * A * B = 3.0
    np.testing.assert_allclose(rates, np.array([3.0]))
    rhs_dict = array_to_species_dict(compiled, rhs)

    assert rhs_dict["A"] == pytest.approx(-3.0)
    assert rhs_dict["B"] == pytest.approx(-3.0)
    assert rhs_dict["C"] == pytest.approx(3.0)


def test_parameter_dict_to_array_rejects_missing_parameter():
    model = build_model_spec("A>B")
    compiled = compile_mass_action_model(model)

    with pytest.raises(ValueError, match="Missing parameter values"):
        parameter_dict_to_array(compiled, {})


def test_concentration_dict_to_array_rejects_missing_species():
    model = build_model_spec("A>B")
    compiled = compile_mass_action_model(model)

    with pytest.raises(ValueError, match="Missing concentration values"):
        concentration_dict_to_array(
            compiled,
            {
                "A": 1.0,
            },
        )


def test_evaluate_rhs_rejects_wrong_concentration_shape():
    model = build_model_spec("A>B")
    compiled = compile_mass_action_model(model)

    with pytest.raises(ValueError, match="concentrations must have shape"):
        evaluate_mass_action_rhs(
            compiled_model=compiled,
            concentrations=np.array([1.0]),
            parameters=np.array([0.1]),
        )


def test_evaluate_rhs_rejects_wrong_parameter_shape():
    model = build_model_spec("A>B")
    compiled = compile_mass_action_model(model)

    with pytest.raises(ValueError, match="parameters must have shape"):
        evaluate_mass_action_rhs(
            compiled_model=compiled,
            concentrations=np.array([1.0, 0.0]),
            parameters=np.array([0.1, 0.2]),
        )


def test_clip_negative_concentrations():
    model = build_model_spec("A>B")
    compiled = compile_mass_action_model(model)

    rates = evaluate_mass_action_rates(
        compiled_model=compiled,
        concentrations=np.array([-2.0, 0.0]),
        parameters=np.array([0.5]),
        clip_negative_concentrations=True,
    )

    np.testing.assert_allclose(rates, np.array([0.0]))
