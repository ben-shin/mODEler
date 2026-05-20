import numpy as np
import pandas as pd

from odefit.data.dataset import Dataset
from odefit.fitting.fit_settings import FitSettings
from odefit.fitting.initial_condition_spec import InitialConditionSpec
from odefit.fitting.multispecies_variable_projection_profile_likelihood import (
    fit_multispecies_variable_projection_profile_likelihood,
)
from odefit.fitting.parameter_spec import ParameterSpec
from odefit.model.model_spec import build_model_spec


def test_multispecies_variable_projection_profile_likelihood_runs():
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

    result = fit_multispecies_variable_projection_profile_likelihood(
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
        profile_parameters=["k1f"],
        n_points=5,
        show_progress=False,
    )

    assert len(result.profile_table) == 5
    assert "delta_rss" in result.profile_table.columns
    assert result.original_result.success
