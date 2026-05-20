import pytest
import numpy as np
import pandas as pd

from odefit.api.backend import (
    bootstrap_global_observables_from_config,
    compare_global_observables_from_config,
    fit_global_observables_from_config,
    profile_likelihood_global_observables_from_config,
)


def _make_data_file(tmp_path):
    time = np.linspace(0.0, 5.0, 12)
    signal = np.exp(-0.4 * time)

    dataframe = pd.DataFrame(
        {
            "time": time,
            "peak_0": signal + 0.01,
            "peak_1": 2.0 * signal - 0.02,
        }
    )

    path = tmp_path / "data.csv"
    dataframe.to_csv(path, index=False)

    return path


def _single_species_config(data_path):
    return {
        "engine_name": "not_a_real_engine",
        "model_text": "A -> B",
        "data": str(data_path),
        "time_column": "time",
        "observed_species": "A",
        "use_variable_projection": True,
        "parameters": {
            "k1f": {
                "initial_guess": 0.2,
                "lower_bound": 1e-6,
                "upper_bound": 10.0,
            }
        },
        "initial_conditions": {
            "A": {"value": 1.0, "mode": "fixed"},
            "B": {"value": 0.0, "mode": "fixed"},
        },
        "max_nfev": 50,
        "show_progress": False,
    }


def _multispecies_config(data_path):
    config = _single_species_config(data_path)

    config.update(
        {
            "use_variable_projection": False,
            "use_multispecies_variable_projection": True,
            "observed_species": ["A", "B"],
            "fit_offset": True,
        }
    )

    return config


def test_fit_api_propagates_engine_name_single_species(tmp_path):
    data_path = _make_data_file(tmp_path)

    with pytest.raises(ValueError, match="Unknown engine bundle"):
        fit_global_observables_from_config(
            _single_species_config(data_path)
        )


def test_fit_api_propagates_engine_name_multispecies(tmp_path):
    data_path = _make_data_file(tmp_path)

    with pytest.raises(ValueError, match="Unknown engine bundle"):
        fit_global_observables_from_config(
            _multispecies_config(data_path)
        )


def test_compare_api_propagates_engine_name(tmp_path):
    data_path = _make_data_file(tmp_path)

    config = {
        "engine_name": "not_a_real_engine",
        "use_variable_projection": True,
        "model_texts": {
            "single_step": "A -> B",
            "two_step": "A -> B\nB -> C",
        },
        "data": str(data_path),
        "time_column": "time",
        "observed_species": "A",
        "parameters_by_model": {
            "single_step": {
                "k1f": {
                    "initial_guess": 0.2,
                    "lower_bound": 1e-6,
                    "upper_bound": 10.0,
                }
            },
            "two_step": {
                "k1f": {
                    "initial_guess": 0.2,
                    "lower_bound": 1e-6,
                    "upper_bound": 10.0,
                },
                "k2f": {
                    "initial_guess": 0.1,
                    "lower_bound": 1e-6,
                    "upper_bound": 10.0,
                },
            },
        },
        "initial_conditions_by_model": {
            "single_step": {
                "A": {"value": 1.0, "mode": "fixed"},
                "B": {"value": 0.0, "mode": "fixed"},
            },
            "two_step": {
                "A": {"value": 1.0, "mode": "fixed"},
                "B": {"value": 0.0, "mode": "fixed"},
                "C": {"value": 0.0, "mode": "fixed"},
            },
        },
        "max_nfev": 50,
        "show_progress": False,
    }

    with pytest.raises(RuntimeError, match="Unknown engine bundle"):
        compare_global_observables_from_config(config)


def test_bootstrap_api_propagates_engine_name(tmp_path):
    data_path = _make_data_file(tmp_path)

    config = _single_species_config(data_path)
    config.update(
        {
            "n_bootstrap": 2,
            "n_workers": 1,
            "show_progress": False,
        }
    )

    with pytest.raises(ValueError, match="Unknown engine bundle"):
        bootstrap_global_observables_from_config(config)


def test_profile_likelihood_api_propagates_engine_name(tmp_path):
    data_path = _make_data_file(tmp_path)

    config = _single_species_config(data_path)
    config.update(
        {
            "profile_parameters": ["k1f"],
            "profile_n_points": 3,
            "show_progress": False,
        }
    )

    with pytest.raises(ValueError, match="Unknown engine bundle"):
        profile_likelihood_global_observables_from_config(config)

def test_multistart_model_comparison_api_propagates_engine_name(tmp_path):
    from odefit.fitting.variable_projection_multistart_model_comparison import (
        fit_global_observable_variable_projection_multistart_model_comparison,
    )
    from odefit.fitting.fit_settings import FitSettings
    from odefit.fitting.initial_condition_spec import InitialConditionSpec
    from odefit.fitting.parameter_spec import ParameterSpec
    from odefit.model.model_spec import build_model_spec
    from odefit.data.dataset import Dataset

    data_path = _make_data_file(tmp_path)
    dataframe = pd.read_csv(data_path)

    dataset = Dataset(
        raw_dataframe=dataframe,
        time_column="time",
        signal_columns=["peak_0", "peak_1"],
    )

    models = {
        "single_step": build_model_spec("A -> B", name="single_step"),
    }

    parameter_specs_by_model = {
        "single_step": [
            ParameterSpec(
                name="k1f",
                initial_guess=0.2,
                lower_bound=1e-6,
                upper_bound=10.0,
            )
        ]
    }

    initial_condition_specs_by_model = {
        "single_step": [
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
    }

    with pytest.raises(RuntimeError, match="Unknown engine bundle"):
        fit_global_observable_variable_projection_multistart_model_comparison(
            models=models,
            dataset=dataset,
            parameter_specs_by_model=parameter_specs_by_model,
            initial_condition_specs_by_model=initial_condition_specs_by_model,
            observed_species_by_model="A",
            settings=FitSettings(species_mapping={}),
            signal_columns=dataset.signal_columns,
            fit_scale=True,
            fit_offset=True,
            backend="numpy",
            method="LSODA",
            n_starts=2,
            random_seed=123,
            sort_by="bic",
            multistart_sort_by="bic",
            show_progress=False,
            engine_name="not_a_real_engine",
        )
