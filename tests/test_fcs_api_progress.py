import numpy as np
import pandas as pd

from odefit.api.fcs import (
    compare_fcs_models_from_config,
    fit_fcs_model_from_config,
)


def _make_fcs_data(tmp_path):
    tau = np.linspace(0.0, 5.0, 16)
    signal = np.exp(-0.4 * tau)

    dataframe = pd.DataFrame(
        {
            "tau": tau,
            "fcs_trace_0": 1.2 * signal + 0.05,
            "fcs_trace_1": 0.8 * signal - 0.02,
        }
    )

    path = tmp_path / "fcs_data.csv"
    dataframe.to_csv(path, index=False)

    return path


def _base_config(data_path):
    return {
        "engine_name": "reference",
        "data": str(data_path),
        "time_column": "tau",
        "signal_columns": ["fcs_trace_0", "fcs_trace_1"],
        "observed_species": "A",
        "use_variable_projection": True,
        "fit_scale": True,
        "fit_offset": True,
        "parameters": {
            "k1f": {
                "initial_guess": 0.2,
                "lower_bound": 1e-6,
                "upper_bound": 10.0,
            }
        },
        "initial_conditions": {
            "A": {
                "value": 1.0,
                "mode": "fixed",
            },
            "B": {
                "value": 0.0,
                "mode": "fixed",
            },
        },
        "method": "trf",
        "loss": "linear",
        "max_nfev": 60,
        "rtol": 1e-6,
        "atol": 1e-9,
        "variable_projection_backend": "numpy",
        "variable_projection_method": "LSODA",
        "show_progress": False,
        "no_plots": True,
    }


def test_fit_fcs_model_from_config_emits_progress(tmp_path):
    data_path = _make_fcs_data(tmp_path)

    config = _base_config(data_path)
    config["model_text"] = "A -> B"

    events = []

    output = fit_fcs_model_from_config(
        config,
        progress_callback=events.append,
    )

    assert output["result"].success
    assert len(events) == 2
    assert events[0].current == 0
    assert events[-1].current == 1
    assert events[-1].eta_seconds == 0.0


def test_compare_fcs_models_from_config_emits_progress_and_table(tmp_path):
    data_path = _make_fcs_data(tmp_path)

    config = _base_config(data_path)
    config["model_texts"] = {
        "single_step": "A -> B",
        "two_step": "A -> B\nB -> C",
    }

    config["parameters_by_model"] = {
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
    }

    config["initial_conditions_by_model"] = {
        "single_step": {
            "A": {
                "value": 1.0,
                "mode": "fixed",
            },
            "B": {
                "value": 0.0,
                "mode": "fixed",
            },
        },
        "two_step": {
            "A": {
                "value": 1.0,
                "mode": "fixed",
            },
            "B": {
                "value": 0.0,
                "mode": "fixed",
            },
            "C": {
                "value": 0.0,
                "mode": "fixed",
            },
        },
    }

    events = []

    output = compare_fcs_models_from_config(
        config,
        progress_callback=events.append,
    )

    table = output["comparison_table"]

    assert len(table) == 2
    assert set(table["model_name"]) == {"single_step", "two_step"}
    assert "rank" in table.columns

    assert events
    assert events[0].current == 0
    assert events[-1].current == 2
    assert events[-1].total == 2

    messages = [event.message for event in events]

    assert any("Fitting model single_step" in message for message in messages)
    assert any("Fitting model two_step" in message for message in messages)
