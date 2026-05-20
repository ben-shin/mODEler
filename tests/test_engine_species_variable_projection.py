import numpy as np
import pandas as pd

from odefit.data.dataset import Dataset
from odefit.engines.registry import get_engine_bundle
from odefit.fitting.fit_settings import FitSettings
from odefit.fitting.initial_condition_spec import InitialConditionSpec
from odefit.fitting.parameter_spec import ParameterSpec
from odefit.fitting.variable_projection import (
    fit_global_observable_model_variable_projection,
    project_observables_onto_species,
)
from odefit.model.model_spec import build_model_spec


def _make_dataset():
    time = np.linspace(0.0, 5.0, 15)
    signal = np.exp(-0.4 * time)

    dataframe = pd.DataFrame(
        {
            "time": time,
            "peak_0": signal + 0.01,
            "peak_1": 2.0 * signal - 0.02,
        }
    )

    return Dataset(
        raw_dataframe=dataframe,
        time_column="time",
        signal_columns=["peak_0", "peak_1"],
    )


def test_project_observables_uses_reference_projection_engine():
    dataset = _make_dataset()
    engine_bundle = get_engine_bundle("reference")

    timepoints = dataset.raw_dataframe["time"].to_numpy(dtype=float)
    species_values = np.exp(-0.4 * timepoints)

    result = project_observables_onto_species(
        timepoints=timepoints,
        simulated_species_values=species_values,
        observed_dataframe=dataset.raw_dataframe,
        signal_columns=dataset.signal_columns,
        fit_scale=True,
        fit_offset=True,
        engine_bundle=engine_bundle,
    )

    assert result.observable_table.shape[0] == 2
    assert "scale" in result.observable_table.columns
    assert "offset" in result.observable_table.columns
    assert result.rss >= 0.0
    assert result.predicted_dataframe.shape[0] == len(timepoints)
    assert result.residuals_dataframe.shape[0] == len(timepoints)


def test_single_species_variable_projection_fit_uses_reference_engine():
    dataset = _make_dataset()

    model = build_model_spec(
        "A -> B",
        name="single_step",
    )

    result = fit_global_observable_model_variable_projection(
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
        observed_species="A",
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
        fit_scale=True,
        fit_offset=True,
        engine_name="reference",
    )

    assert result.success
    assert "k1f" in result.fitted_parameters
    assert result.statistics["rss"] >= 0.0
    assert result.nfev > 0
    assert result.active_mask is not None
