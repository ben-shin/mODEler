import numpy as np
import pandas as pd

from odefit.api.backend import (
    bootstrap_global_observables_from_config,
    compare_global_observables_from_config,
    fit_global_observables_from_config,
    profile_likelihood_global_observables_from_config,
)


def _make_data_file(tmp_path):
    time = np.linspace(0.0, 5.0, 15)
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
        "engine_name": "reference",
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
        "max_nfev": 100,
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

    config.pop("fit_scale", None)

    return config


def test_reference_engine_single_species_fit_from_api(tmp_path):
    data_path = _make_data_file(tmp_path)

    output = fit_global_observables_from_config(
        _single_species_config(data_path)
    )

    assert output["result"].success
    assert "k1f" in output["result"].fitted_parameters


def test_reference_engine_multispecies_fit_from_api(tmp_path):
    data_path = _make_data_file(tmp_path)

    output = fit_global_observables_from_config(
        _multispecies_config(data_path)
    )

    assert output["result"].success
    assert "k1f" in output["result"].fitted_parameters


def test_reference_engine_model_comparison_from_api(tmp_path):
    data_path = _make_data_file(tmp_path)

    config = {
        "engine_name": "reference",
        "model_texts": {
            "single_step": "A -> B",
            "two_step": "A -> B\nB -> C",
        },
        "data": str(data_path),
        "time_column": "time",
        "observed_species": "A",
        "use_variable_projection": True,
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
        "max_nfev": 100,
        "show_progress": False,
    }

    output = compare_global_observables_from_config(config)

    assert output["result"].best_model_name in {"single_step", "two_step"}


def test_reference_engine_bootstrap_from_api(tmp_path):
    data_path = _make_data_file(tmp_path)

    config = _single_species_config(data_path)
    config.update(
        {
            "n_bootstrap": 2,
            "n_workers": 1,
            "show_progress": False,
        }
    )

    output = bootstrap_global_observables_from_config(config)

    assert output["result"].parameter_samples.shape[0] == 2


def test_reference_engine_profile_likelihood_from_api(tmp_path):
    data_path = _make_data_file(tmp_path)

    config = _single_species_config(data_path)
    config.update(
        {
            "profile_parameters": ["k1f"],
            "profile_n_points": 3,
            "show_progress": False,
        }
    )

    output = profile_likelihood_global_observables_from_config(config)

    assert len(output["result"].profile_table) == 3
