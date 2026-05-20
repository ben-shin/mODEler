import json

import numpy as np
import pandas as pd

from odefit.api.backend import (
    bootstrap_global_observables_from_config,
    fit_global_observables_from_config,
    profile_likelihood_global_observables_from_config,
)
from odefit.api.serialization import (
    backend_output_payload,
    dataframe_preview,
    table_payload,
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


def _base_config(data_path):
    return {
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


def test_dataframe_preview_is_json_serializable():
    dataframe = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [0.1, 0.2, 0.3],
        }
    )

    payload = dataframe_preview(dataframe, max_rows=2)

    assert payload["n_rows"] == 3
    assert payload["n_preview_rows"] == 2

    json.dumps(payload)


def test_table_payload_is_json_serializable():
    dataframe = pd.DataFrame(
        {
            "parameter": ["k1f"],
            "value": [0.4],
        }
    )

    payload = table_payload(dataframe)

    assert payload["n_rows"] == 1
    json.dumps(payload)


def test_fit_backend_output_payload_is_json_serializable(tmp_path):
    data_path = _make_data_file(tmp_path)

    output = fit_global_observables_from_config(
        _base_config(data_path)
    )

    payload = backend_output_payload(
        output,
        workflow="fit",
        max_rows=5,
    )

    assert payload["workflow"] == "fit"
    assert payload["result"]["success"]
    assert "k1f" in payload["result"]["fitted_parameters"]

    json.dumps(payload)


def test_bootstrap_backend_output_payload_is_json_serializable(tmp_path):
    data_path = _make_data_file(tmp_path)
    config = _base_config(data_path)
    config["n_bootstrap"] = 2
    config["n_workers"] = 1

    output = bootstrap_global_observables_from_config(config)

    payload = backend_output_payload(
        output,
        workflow="bootstrap",
        max_rows=5,
    )

    assert payload["workflow"] == "bootstrap"
    assert payload["result"]["n_successful_bootstrap"] == 2

    json.dumps(payload)


def test_profile_backend_output_payload_is_json_serializable(tmp_path):
    data_path = _make_data_file(tmp_path)
    config = _base_config(data_path)
    config["profile_parameters"] = ["k1f"]
    config["profile_n_points"] = 3

    output = profile_likelihood_global_observables_from_config(config)

    payload = backend_output_payload(
        output,
        workflow="profile_likelihood",
        max_rows=5,
    )

    assert payload["workflow"] == "profile_likelihood"
    assert payload["result"]["n_profile_points"] == 3

    json.dumps(payload)
