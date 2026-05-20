import numpy as np
import pandas as pd

from odefit.data.dataset import Dataset
from odefit.fitting.fit_settings import FitSettings
from odefit.fitting.initial_condition_spec import InitialConditionSpec
from odefit.fitting.multispecies_variable_projection_multistart_model_comparison import (
    fit_global_observable_multispecies_variable_projection_multistart_model_comparison,
)
from odefit.fitting.parameter_spec import ParameterSpec
from odefit.model.model_spec import build_model_spec


def test_multispecies_variable_projection_multistart_model_comparison_runs():
    time = np.linspace(0.0, 5.0, 20)

    a = np.exp(-0.4 * time)
    b = 1.0 - a

    rng = np.random.default_rng(123)
    dataframe = pd.DataFrame({"time": time})

    for i in range(4):
        dataframe[f"peak_{i}"] = (
            rng.uniform(0.5, 2.0) * a
            + rng.uniform(-1.0, 1.0) * b
            + rng.uniform(-0.1, 0.1)
            + rng.normal(0.0, 0.002, size=len(time))
        )

    dataset = Dataset(
        raw_dataframe=dataframe,
        time_column="time",
        signal_columns=[f"peak_{i}" for i in range(4)],
    )

    models = {
        "single_step": build_model_spec("A -> B", name="single_step"),
        "two_step": build_model_spec("A -> B\nB -> C", name="two_step"),
    }

    parameter_specs_by_model = {
        "single_step": [
            ParameterSpec("k1f", 0.2, 1e-6, 10.0),
        ],
        "two_step": [
            ParameterSpec("k1f", 0.2, 1e-6, 10.0),
            ParameterSpec("k2f", 0.1, 1e-6, 10.0),
        ],
    }

    initial_condition_specs_by_model = {
        "single_step": [
            InitialConditionSpec("A", 1.0, fixed=True, fixed_value=1.0),
            InitialConditionSpec("B", 0.0, fixed=True, fixed_value=0.0),
        ],
        "two_step": [
            InitialConditionSpec("A", 1.0, fixed=True, fixed_value=1.0),
            InitialConditionSpec("B", 0.0, fixed=True, fixed_value=0.0),
            InitialConditionSpec("C", 0.0, fixed=True, fixed_value=0.0),
        ],
    }

    result = (
        fit_global_observable_multispecies_variable_projection_multistart_model_comparison(
            models=models,
            dataset=dataset,
            parameter_specs_by_model=parameter_specs_by_model,
            initial_condition_specs_by_model=initial_condition_specs_by_model,
            observed_species_by_model={
                "single_step": ["A", "B"],
                "two_step": ["A", "B"],
            },
            settings=FitSettings(
                species_mapping={},
                use_normalized_data=False,
                method="trf",
                loss="linear",
                max_nfev=100,
                rtol=1e-6,
                atol=1e-9,
            ),
            signal_columns=dataset.signal_columns,
            n_starts=3,
            random_seed=123,
            sort_by="bic",
            multistart_sort_by="bic",
            show_progress=False,
        )
    )

    assert result.best_model_name in models
    assert result.best_fit_result.success
    assert len(result.comparison_table) == 2
    assert "bic" in result.comparison_table.columns
    assert list(result.comparison_table["rank"]) == [1, 2]
