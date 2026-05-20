import numpy as np
import pandas as pd

from odefit.data.dataset import Dataset
from odefit.fitting.fit_settings import FitSettings
from odefit.fitting.initial_condition_spec import InitialConditionSpec
from odefit.fitting.multispecies_variable_projection_bootstrap import (
    bootstrap_global_observable_multispecies_variable_projection_fit,
)
from odefit.fitting.parameter_spec import ParameterSpec
from odefit.model.model_spec import build_model_spec


def test_multispecies_variable_projection_bootstrap_runs():
    time = np.linspace(0.0, 5.0, 15)
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

    result = bootstrap_global_observable_multispecies_variable_projection_fit(
        model=build_model_spec("A -> B", name="single_step"),
        dataset=dataset,
        parameter_specs=[
            ParameterSpec("k1f", 0.2, 1e-6, 10.0),
        ],
        initial_condition_specs=[
            InitialConditionSpec("A", 1.0, fixed=True, fixed_value=1.0),
            InitialConditionSpec("B", 0.0, fixed=True, fixed_value=0.0),
        ],
        observed_species=["A", "B"],
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
        n_bootstrap=3,
        n_workers=1,
        random_seed=123,
        show_progress=False,
    )

    assert len(result.bootstrap_results) == 3
    assert result.parameter_samples.shape[0] == 3
    assert "k1f" in result.parameter_samples.columns
    assert "ci_lower" in result.summary_table.columns
