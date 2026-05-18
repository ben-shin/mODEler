import numpy as np
import pandas as pd
import pytest

from odefit.data.dataset import Dataset
from odefit.fitting.fit_settings import FitSettings
from odefit.fitting.initial_condition_spec import InitialConditionSpec
from odefit.fitting.observable_spec import ObservableSpec
from odefit.fitting.optimizer import fit_model
from odefit.fitting.parameter_spec import ParameterSpec
from odefit.model.model_spec import build_model_spec


def test_fit_observable_scale_and_offset_synthetic_data():
    model = build_model_spec("A>B")

    true_k = 0.5
    true_scale = 2.0
    true_offset = 0.1

    timepoints = np.linspace(0.0, 5.0, 40)

    a_values = np.exp(-true_k * timepoints)
    amide_signal = true_scale * a_values + true_offset

    dataset = Dataset(
        raw_dataframe=pd.DataFrame(
            {
                "time": timepoints,
                "amide": amide_signal,
            }
        ),
        time_column="time",
        signal_columns=["amide"],
    )

    parameter_specs = [
        ParameterSpec(
            name="k1f",
            initial_guess=0.2,
            lower_bound=0.0,
            upper_bound=10.0,
        )
    ]

    initial_condition_specs = [
        InitialConditionSpec(
            species="A",
            initial_guess=1.0,
            fixed=True,
            fixed_value=1.0,
        ),
        InitialConditionSpec(
            species="B",
            initial_guess=0.0,
            fixed=True,
            fixed_value=0.0,
        ),
    ]

    observable_specs = [
        ObservableSpec(
            data_column="amide",
            species="A",
            scale_initial_guess=1.0,
            scale_lower_bound=0.0,
            scale_upper_bound=10.0,
            scale_fixed=False,
            offset_initial_guess=0.0,
            offset_lower_bound=-1.0,
            offset_upper_bound=1.0,
            offset_fixed=False,
        )
    ]

    result = fit_model(
        model=model,
        dataset=dataset,
        parameter_specs=parameter_specs,
        initial_condition_specs=initial_condition_specs,
        observable_specs=observable_specs,
        settings=FitSettings(
            species_mapping={},
            rtol=1e-9,
            atol=1e-11,
        ),
    )

    assert result.success

    assert result.fitted_parameters["k1f"] == pytest.approx(true_k, rel=1e-2)

    assert result.fitted_observables is not None

    assert result.fitted_observables["amide"]["scale"] == pytest.approx(
        true_scale,
        rel=1e-2,
    )

    assert result.fitted_observables["amide"]["offset"] == pytest.approx(
        true_offset,
        abs=1e-2,
    )

    assert result.statistics["rmse"] < 1e-3

    # k1f, scale, offset are free.
    assert result.statistics["n_parameters"] == 3.0


def test_fit_observable_fixed_scale_free_offset():
    model = build_model_spec("A>B")

    true_k = 0.5
    true_offset = 0.25

    timepoints = np.linspace(0.0, 5.0, 30)

    a_values = np.exp(-true_k * timepoints)
    amide_signal = a_values + true_offset

    dataset = Dataset(
        raw_dataframe=pd.DataFrame(
            {
                "time": timepoints,
                "amide": amide_signal,
            }
        ),
        time_column="time",
        signal_columns=["amide"],
    )

    parameter_specs = [
        ParameterSpec(
            name="k1f",
            initial_guess=0.2,
            lower_bound=0.0,
            upper_bound=10.0,
        )
    ]

    initial_condition_specs = [
        InitialConditionSpec("A", initial_guess=1.0, fixed=True, fixed_value=1.0),
        InitialConditionSpec("B", initial_guess=0.0, fixed=True, fixed_value=0.0),
    ]

    observable_specs = [
        ObservableSpec(
            data_column="amide",
            species="A",
            scale_fixed=True,
            scale_fixed_value=1.0,
            offset_initial_guess=0.0,
            offset_lower_bound=-1.0,
            offset_upper_bound=1.0,
            offset_fixed=False,
        )
    ]

    result = fit_model(
        model=model,
        dataset=dataset,
        parameter_specs=parameter_specs,
        initial_condition_specs=initial_condition_specs,
        observable_specs=observable_specs,
        settings=FitSettings(
            species_mapping={},
            rtol=1e-9,
            atol=1e-11,
        ),
    )

    assert result.success

    assert result.fitted_parameters["k1f"] == pytest.approx(true_k, rel=1e-2)

    assert result.fitted_observables is not None
    assert result.fitted_observables["amide"]["scale"] == 1.0
    assert result.fitted_observables["amide"]["offset"] == pytest.approx(
        true_offset,
        abs=1e-2,
    )

    # k1f and offset are free.
    assert result.statistics["n_parameters"] == 2.0


def test_old_direct_species_mapping_still_works():
    model = build_model_spec("A>B")

    true_k = 0.5
    timepoints = np.linspace(0.0, 5.0, 30)

    a_values = np.exp(-true_k * timepoints)
    b_values = 1.0 - a_values

    dataset = Dataset(
        raw_dataframe=pd.DataFrame(
            {
                "time": timepoints,
                "A": a_values,
                "B": b_values,
            }
        ),
        time_column="time",
        signal_columns=["A", "B"],
    )

    result = fit_model(
        model=model,
        dataset=dataset,
        parameter_specs=[
            ParameterSpec(
                name="k1f",
                initial_guess=0.2,
                lower_bound=0.0,
                upper_bound=10.0,
            )
        ],
        initial_conditions={
            "A": 1.0,
            "B": 0.0,
        },
        settings=FitSettings(
            species_mapping={
                "A": "A",
                "B": "B",
            },
            rtol=1e-9,
            atol=1e-11,
        ),
    )

    assert result.success
    assert result.fitted_parameters["k1f"] == pytest.approx(true_k, rel=1e-2)
    assert result.fitted_observables is None
