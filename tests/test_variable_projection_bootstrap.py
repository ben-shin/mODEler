import numpy as np
import pandas as pd

from odefit.data.dataset import Dataset
from odefit.fitting.fit_settings import FitSettings
from odefit.fitting.initial_condition_spec import InitialConditionSpec
from odefit.fitting.parameter_spec import ParameterSpec
from odefit.fitting.variable_projection_bootstrap import (
    bootstrap_global_observable_variable_projection_fit,
)
from odefit.model.model_spec import build_model_spec


def test_variable_projection_bootstrap_runs():
    time = np.linspace(0.0, 5.0, 12)
    signal = np.exp(-0.4 * time)

    rng = np.random.default_rng(123)

    dataframe = pd.DataFrame({"time": time})

    for i in range(4):
        scale = rng.uniform(0.5, 2.0)
        offset = rng.uniform(-0.1, 0.1)
        noise = rng.normal(0.0, 0.002, size=len(time))

        dataframe[f"peak_{i}"] = scale * signal + offset + noise

    dataset = Dataset(
        raw_dataframe=dataframe,
        time_column="time",
        signal_columns=[f"peak_{i}" for i in range(4)],
    )

    model = build_model_spec(
        "A -> B",
        name="single_step",
    )

    parameter_specs = [
        ParameterSpec(
            name="k1f",
            initial_guess=0.2,
            lower_bound=1e-6,
            upper_bound=10.0,
        )
    ]

    initial_condition_specs = [
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

    settings = FitSettings(
        species_mapping={},
        use_normalized_data=False,
        method="trf",
        loss="linear",
        max_nfev=100,
        rtol=1e-6,
        atol=1e-9,
    )

    result = bootstrap_global_observable_variable_projection_fit(
        model=model,
        dataset=dataset,
        parameter_specs=parameter_specs,
        initial_condition_specs=initial_condition_specs,
        observed_species="A",
        settings=settings,
        signal_columns=dataset.signal_columns,
        n_bootstrap=3,
        random_seed=123,
        show_progress=False,
    )

    assert len(result.bootstrap_results) == 3
    assert result.parameter_samples.shape[0] == 3
    assert "k1f" in result.parameter_samples.columns
    assert "parameter" in result.summary_table.columns
    assert "ci_lower" in result.summary_table.columns
    assert "ci_upper" in result.summary_table.columns
