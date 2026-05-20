import numpy as np
import pandas as pd

from odefit.data.dataset import Dataset
from odefit.fitting.fit_settings import FitSettings
from odefit.fitting.initial_condition_spec import InitialConditionSpec
from odefit.fitting.multispecies_variable_projection_multistart import (
    fit_global_observable_model_multispecies_variable_projection_multistart,
)
from odefit.fitting.parameter_spec import ParameterSpec
from odefit.model.model_spec import build_model_spec


def test_multispecies_variable_projection_multistart_runs():
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

    model = build_model_spec("A -> B", name="single_step")

    result = fit_global_observable_model_multispecies_variable_projection_multistart(
        model=model,
        dataset=dataset,
        parameter_specs=[
            ParameterSpec(
                name="k1f",
                initial_guess=0.2,
                lower_bound=1e-6,
                upper_bound=10.0,
            )
        ],
        initial_condition_specs=[
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
        n_starts=3,
        random_seed=123,
        show_progress=False,
    )

    assert result.best_result.success
    assert result.best_index in {0, 1, 2}
    assert len(result.comparison_table) == 3
    assert "k1f" in result.best_result.fitted_parameters
