import numpy as np
import pandas as pd
import pytest

from odefit.api.backend import fit_global_observables_from_config
from odefit.engines.jax_projection import is_jax_available


def _make_data_file(tmp_path):
    time = np.linspace(0.0, 5.0, 20)
    signal = np.exp(-0.4 * time)

    dataframe = pd.DataFrame(
        {
            "time": time,
            "peak_0": 1.5 * signal + 0.01,
            "peak_1": 0.7 * signal - 0.02,
            "peak_2": 2.0 * signal + 0.1,
        }
    )

    path = tmp_path / "jax_projection_workflow_data.csv"
    dataframe.to_csv(path, index=False)

    return path


def _fit_config(data_path, engine_name):
    return {
        "engine_name": engine_name,
        "model_text": "A -> B",
        "data": str(data_path),
        "time_column": "time",
        "signal_columns": None,
        "exclude_columns": None,
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
        "max_nfev": 100,
        "rtol": 1e-6,
        "atol": 1e-9,
        "variable_projection_backend": "numpy",
        "variable_projection_method": "LSODA",
        "show_progress": False,
        "no_plots": True,
    }


@pytest.mark.skipif(
    not is_jax_available(),
    reason="jax is not installed",
)
def test_jax_projection_engine_runs_through_api_fit(tmp_path):
    data_path = _make_data_file(tmp_path)

    output = fit_global_observables_from_config(
        _fit_config(
            data_path=data_path,
            engine_name="jax_projection",
        )
    )

    result = output["result"]

    assert result.success
    assert "k1f" in result.fitted_parameters
    assert result.statistics["rss"] >= 0.0
    assert result.observable_table.shape[0] == 3


@pytest.mark.skipif(
    not is_jax_available(),
    reason="jax is not installed",
)
def test_jax_projection_engine_matches_reference_workflow_reasonably(tmp_path):
    data_path = _make_data_file(tmp_path)

    reference_output = fit_global_observables_from_config(
        _fit_config(
            data_path=data_path,
            engine_name="reference",
        )
    )

    jax_output = fit_global_observables_from_config(
        _fit_config(
            data_path=data_path,
            engine_name="jax_projection",
        )
    )

    reference_result = reference_output["result"]
    jax_result = jax_output["result"]

    assert reference_result.success
    assert jax_result.success

    assert np.isclose(
        jax_result.fitted_parameters["k1f"],
        reference_result.fitted_parameters["k1f"],
        rtol=1e-5,
        atol=1e-7,
    )

    assert np.isclose(
        jax_result.statistics["rss"],
        reference_result.statistics["rss"],
        rtol=1e-5,
        atol=1e-8,
    )
