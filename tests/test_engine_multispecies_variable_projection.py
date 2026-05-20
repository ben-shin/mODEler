import numpy as np
import pandas as pd

from odefit.data.dataset import Dataset
from odefit.engines.registry import get_engine_bundle
from odefit.fitting.fit_settings import FitSettings
from odefit.fitting.initial_condition_spec import InitialConditionSpec
from odefit.fitting.multispecies_variable_projection import (
    fit_global_observable_model_multispecies_variable_projection,
)
from odefit.fitting.parameter_spec import ParameterSpec
from odefit.model.model_spec import build_model_spec


def _make_dataset():
    time = np.linspace(0.0, 5.0, 20)

    species_a = np.exp(-0.4 * time)
    species_b = 1.0 - species_a

    dataframe = pd.DataFrame(
        {
            "time": time,
            "peak_0": 1.5 * species_a + 0.4 * species_b + 0.01,
            "peak_1": 0.7 * species_a - 0.2 * species_b - 0.02,
        }
    )

    return Dataset(
        raw_dataframe=dataframe,
        time_column="time",
        signal_columns=["peak_0", "peak_1"],
    )


def test_reference_engine_multispecies_projection_directly():
    engine_bundle = get_engine_bundle("reference")

    time = np.linspace(0.0, 5.0, 20)
    species_a = np.exp(-0.4 * time)
    species_b = 1.0 - species_a

    species_matrix = np.column_stack(
        [
            species_a,
            species_b,
        ]
    )

    observed = 1.5 * species_a + 0.4 * species_b + 0.01

    result = engine_bundle.projection.project_multispecies(
        observed_values=observed,
        species_matrix=species_matrix,
        species_names=["A", "B"],
        fit_offset=True,
    )

    assert "A" in result.coefficients
    assert "B" in result.coefficients
    assert result.rss < 1e-20


def test_multispecies_variable_projection_fit_uses_reference_engine():
    dataset = _make_dataset()

    model = build_model_spec(
        "A -> B",
        name="single_step",
    )

    result = fit_global_observable_model_multispecies_variable_projection(
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
        fit_offset=True,
        engine_name="reference",
    )

    assert result.success
    assert "k1f" in result.fitted_parameters
    assert result.statistics["rss"] >= 0.0
    assert result.optimizer_result.nfev > 0
    assert result.observable_table.shape[0] == 2
    assert "coefficient_A" in result.observable_table.columns
    assert "coefficient_B" in result.observable_table.columns
