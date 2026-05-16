import numpy as np
import pandas as pd
import pytest

from odefit.data.dataset import Dataset
from odefit.fitting.fit_settings import FitSettings
from odefit.fitting.optimizer import fit_model
from odefit.fitting.parameter_spec import ParameterSpec
from odefit.model.model_spec import build_model_spec


def test_fit_irreversible_decay_synthetic_data():
    model = build_model_spec("A>B")

    true_k = 0.5

    timepoints = np.linspace(0.0, 5.0, 20)
    a_values = np.exp(-true_k * timepoints)
    b_values = 1.0 - a_values

    dataframe = pd.DataFrame(
        {
            "time": timepoints,
            "A": a_values,
            "B": b_values,
        }
    )

    dataset = Dataset(
        raw_dataframe=dataframe,
        time_column="time",
        signal_columns=["A", "B"],
    )

    parameter_specs = [
        ParameterSpec(
            name="k1f",
            initial_guess=0.1,
            lower_bound=0.0,
            upper_bound=10.0,
        )
    ]

    initial_conditions = {
        "A": 1.0,
        "B": 0.0,
    }

    settings = FitSettings(
        species_mapping={
            "A": "A",
            "B": "B",
        }
    )

    result = fit_model(
        model=model,
        dataset=dataset,
        parameter_specs=parameter_specs,
        initial_conditions=initial_conditions,
        settings=settings,
    )

    assert result.success
    assert result.fitted_parameters["k1f"] == pytest.approx(true_k, rel=1e-2)
    assert result.statistics["rmse"] < 1e-3
