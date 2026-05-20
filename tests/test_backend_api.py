import numpy as np
import pandas as pd

from odefit.api.backend import (
    bootstrap_global_observables_from_config,
    compare_global_observables_from_config,
    fit_global_observables_from_config,
    parse_model_text,
    profile_likelihood_global_observables_from_config,
    simulate_from_text,
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


def test_backend_api_parse_and_simulate():
    parsed = parse_model_text("A -> B", name="test")

    assert "A" in parsed["species"]
    assert "B" in parsed["species"]
    assert "k1f" in parsed["parameters"]

    simulated = simulate_from_text(
        model_text="A -> B",
        parameters={"k1f": 0.4},
        initial_conditions={"A": 1.0, "B": 0.0},
        timepoints=[0.0, 1.0, 2.0],
    )

    assert "A" in simulated.columns
    assert "B" in simulated.columns


def test_backend_api_fit_global_observables(tmp_path):
    data_path = _make_data_file(tmp_path)

    config = {
        "model_text": "A -> B",
        "data": str(data_path),
        "time_column": "time",
        "observed_species": "A",
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

    output = fit_global_observables_from_config(config)

    assert output["result"].success
    assert "k1f" in output["result"].fitted_parameters


def test_backend_api_compare_global_observables(tmp_path):
    data_path = _make_data_file(tmp_path)

    config = {
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
        "max_nfev": 100,
        "show_progress": False,
    }

    output = compare_global_observables_from_config(config)

    assert output["result"].best_model_name in {"single_step", "two_step"}


def test_backend_api_bootstrap_global_observables(tmp_path):
    data_path = _make_data_file(tmp_path)

    config = {
        "model_text": "A -> B",
        "data": str(data_path),
        "time_column": "time",
        "observed_species": "A",
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
        "n_bootstrap": 2,
        "n_workers": 1,
        "max_nfev": 100,
        "show_progress": False,
    }

    output = bootstrap_global_observables_from_config(config)

    assert output["result"].parameter_samples.shape[0] == 2


def test_backend_api_profile_likelihood_global_observables(tmp_path):
    data_path = _make_data_file(tmp_path)

    config = {
        "model_text": "A -> B",
        "data": str(data_path),
        "time_column": "time",
        "observed_species": "A",
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
        "profile_parameters": ["k1f"],
        "profile_n_points": 3,
        "max_nfev": 100,
        "show_progress": False,
    }

    output = profile_likelihood_global_observables_from_config(config)

    assert len(output["result"].profile_table) == 3
