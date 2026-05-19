import numpy as np
import pandas as pd

from odefit.data.dataset import Dataset
from odefit.fitting.fit_settings import FitSettings
from odefit.fitting.initial_condition_spec import InitialConditionSpec
from odefit.fitting.parameter_spec import ParameterSpec
from odefit.fitting.variable_projection_multistart_model_comparison import (
    fit_global_observable_variable_projection_multistart_model_comparison,
)
from odefit.model.model_spec import build_model_spec


def _make_dataset():
    time = np.linspace(0.0, 5.0, 12)
    signal = np.exp(-0.4 * time)

    rng = np.random.default_rng(123)

    dataframe = pd.DataFrame({"time": time})

    for i in range(4):
        scale = rng.uniform(0.5, 2.0)
        offset = rng.uniform(-0.1, 0.1)
        noise = rng.normal(0.0, 0.002, size=len(time))

        dataframe[f"peak_{i}"] = scale * signal + offset + noise

    return Dataset(
        raw_dataframe=dataframe,
        time_column="time",
        signal_columns=[f"peak_{i}" for i in range(4)],
    )


def test_variable_projection_multistart_model_comparison_runs():
    dataset = _make_dataset()

    models = {
        "single_step": build_model_spec(
            "A -> B",
            name="single_step",
        ),
        "two_step": build_model_spec(
            "A -> B\nB -> C",
            name="two_step",
        ),
    }

    parameter_specs_by_model = {
        "single_step": [
            ParameterSpec(
                name="k1f",
                initial_guess=0.2,
                lower_bound=1e-6,
                upper_bound=10.0,
            ),
        ],
        "two_step": [
            ParameterSpec(
                name="k1f",
                initial_guess=0.2,
                lower_bound=1e-6,
                upper_bound=10.0,
            ),
            ParameterSpec(
                name="k2f",
                initial_guess=0.1,
                lower_bound=1e-6,
                upper_bound=10.0,
            ),
        ],
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
        ],
        "two_step": [
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
            InitialConditionSpec(
                species="C",
                initial_guess=0.0,
                fixed=True,
                fixed_value=0.0,
            ),
        ],
    }

    settings = FitSettings(
        species_mapping={},
        use_normalized_data=False,
        method="trf",
        loss="linear",
        max_nfev=100,
        rtol=1e-6,
        atol=1e-9,
    )

    result = fit_global_observable_variable_projection_multistart_model_comparison(
        models=models,
        dataset=dataset,
        parameter_specs_by_model=parameter_specs_by_model,
        initial_condition_specs_by_model=initial_condition_specs_by_model,
        observed_species_by_model="A",
        settings=settings,
        signal_columns=dataset.signal_columns,
        n_starts=3,
        random_seed=123,
        sort_by="bic",
        multistart_sort_by="bic",
        show_progress=False,
    )

    assert result.best_model_name in models
    assert result.best_fit_result.success
    assert len(result.comparison_table) == 2

    for column in ["rank", "model_name", "rss", "rmse", "aic", "bic"]:
        assert column in result.comparison_table.columns
