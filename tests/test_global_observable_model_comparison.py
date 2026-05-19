import numpy as np
import pandas as pd
import pytest

from odefit.data.dataset import Dataset
from odefit.fitting.fit_settings import FitSettings
from odefit.fitting.global_observable_model_comparison import (
    build_model_specs_from_texts,
    export_global_observable_model_comparison,
    fit_global_observable_model_comparison,
)
from odefit.fitting.initial_condition_spec import InitialConditionSpec
from odefit.fitting.parameter_spec import ParameterSpec


def make_global_observable_dataset() -> Dataset:
    true_k = 0.4
    timepoints = np.linspace(0.0, 8.0, 30)
    a_values = np.exp(-true_k * timepoints)

    dataframe = pd.DataFrame(
        {
            "time": timepoints,
            "A23_HN": 2.0 * a_values + 0.10,
            "G45_HN": 1.5 * a_values + 0.20,
            "L78_HN": 0.8 * a_values + 0.05,
        }
    )

    return Dataset(
        raw_dataframe=dataframe,
        time_column="time",
        signal_columns=["A23_HN", "G45_HN", "L78_HN"],
    )


def make_initial_condition_specs() -> list[InitialConditionSpec]:
    return [
        InitialConditionSpec(
            species="A",
            initial_guess=1.0,
            lower_bound=0.0,
            upper_bound=2.0,
            fixed=True,
            fixed_value=1.0,
        ),
        InitialConditionSpec(
            species="B",
            initial_guess=0.0,
            lower_bound=0.0,
            upper_bound=2.0,
            fixed=True,
            fixed_value=0.0,
        ),
    ]


def test_build_model_specs_from_texts():
    models = build_model_specs_from_texts(
        {
            "irreversible": "A>B",
            "reversible": "A-B",
        }
    )

    assert set(models) == {"irreversible", "reversible"}
    assert models["irreversible"].name == "irreversible"
    assert models["reversible"].name == "reversible"
    assert models["irreversible"].parameters == ["k1f"]
    assert models["reversible"].parameters == ["k1f", "k1r"]


def test_fit_global_observable_model_comparison():
    dataset = make_global_observable_dataset()

    models = build_model_specs_from_texts(
        {
            "irreversible": "A>B",
            "reversible": "A-B",
        }
    )

    parameter_specs_by_model = {
        "irreversible": [
            ParameterSpec(
                name="k1f",
                initial_guess=0.1,
                lower_bound=0.001,
                upper_bound=10.0,
            )
        ],
        "reversible": [
            ParameterSpec(
                name="k1f",
                initial_guess=0.1,
                lower_bound=0.001,
                upper_bound=10.0,
            ),
            ParameterSpec(
                name="k1r",
                initial_guess=0.01,
                lower_bound=0.000001,
                upper_bound=10.0,
            ),
        ],
    }

    initial_condition_specs_by_model = {
        "irreversible": make_initial_condition_specs(),
        "reversible": make_initial_condition_specs(),
    }

    result = fit_global_observable_model_comparison(
        models=models,
        dataset=dataset,
        parameter_specs_by_model=parameter_specs_by_model,
        initial_condition_specs_by_model=initial_condition_specs_by_model,
        observed_species_by_model="A",
        settings_by_model=FitSettings(
            species_mapping={},
            rtol=1e-8,
            atol=1e-10,
        ),
        scale_initial_guess=1.0,
        scale_lower_bound=0.0,
        scale_upper_bound=5.0,
        offset_initial_guess=0.0,
        offset_lower_bound=-1.0,
        offset_upper_bound=1.0,
        sort_by="aic",
    )

    assert set(result.fit_results) == {"irreversible", "reversible"}
    assert result.failures == []

    assert result.best_model_name in {"irreversible", "reversible"}
    assert result.best_fit_result.success

    assert "model" in result.comparison_table.columns
    assert "rank" in result.comparison_table.columns
    assert len(result.comparison_table) == 2

    assert set(result.comparison_table["model"]) == {
        "irreversible",
        "reversible",
    }


def test_fit_global_observable_model_comparison_records_failure():
    dataset = make_global_observable_dataset()

    models = build_model_specs_from_texts(
        {
            "valid": "A>B",
            "invalid_observed_species": "A>B",
        }
    )

    parameter_specs_by_model = {
        "valid": [
            ParameterSpec(
                name="k1f",
                initial_guess=0.1,
                lower_bound=0.001,
                upper_bound=10.0,
            )
        ],
        "invalid_observed_species": [
            ParameterSpec(
                name="k1f",
                initial_guess=0.1,
                lower_bound=0.001,
                upper_bound=10.0,
            )
        ],
    }

    initial_condition_specs_by_model = {
        "valid": make_initial_condition_specs(),
        "invalid_observed_species": make_initial_condition_specs(),
    }

    result = fit_global_observable_model_comparison(
        models=models,
        dataset=dataset,
        parameter_specs_by_model=parameter_specs_by_model,
        initial_condition_specs_by_model=initial_condition_specs_by_model,
        observed_species_by_model={
            "valid": "A",
            "invalid_observed_species": "missing",
        },
        settings_by_model=FitSettings(
            species_mapping={},
            rtol=1e-8,
            atol=1e-10,
        ),
        scale_initial_guess=1.0,
        scale_lower_bound=0.0,
        scale_upper_bound=5.0,
        offset_initial_guess=0.0,
        offset_lower_bound=-1.0,
        offset_upper_bound=1.0,
    )

    assert set(result.fit_results) == {"valid"}
    assert len(result.failures) == 1
    assert result.failures[0].model_name == "invalid_observed_species"
    assert result.best_fit_result.success


def test_fit_global_observable_model_comparison_raises_if_all_fail():
    dataset = make_global_observable_dataset()

    models = build_model_specs_from_texts(
        {
            "bad": "A>B",
        }
    )

    parameter_specs_by_model = {
        "bad": [
            ParameterSpec(
                name="k1f",
                initial_guess=0.1,
                lower_bound=0.001,
                upper_bound=10.0,
            )
        ],
    }

    initial_condition_specs_by_model = {
        "bad": make_initial_condition_specs(),
    }

    with pytest.raises(RuntimeError):
        fit_global_observable_model_comparison(
            models=models,
            dataset=dataset,
            parameter_specs_by_model=parameter_specs_by_model,
            initial_condition_specs_by_model=initial_condition_specs_by_model,
            observed_species_by_model="missing",
            settings_by_model=FitSettings(
                species_mapping={},
            ),
        )


def test_export_global_observable_model_comparison(tmp_path):
    dataset = make_global_observable_dataset()

    models = build_model_specs_from_texts(
        {
            "irreversible": "A>B",
            "reversible": "A-B",
        }
    )

    parameter_specs_by_model = {
        "irreversible": [
            ParameterSpec(
                name="k1f",
                initial_guess=0.1,
                lower_bound=0.001,
                upper_bound=10.0,
            )
        ],
        "reversible": [
            ParameterSpec(
                name="k1f",
                initial_guess=0.1,
                lower_bound=0.001,
                upper_bound=10.0,
            ),
            ParameterSpec(
                name="k1r",
                initial_guess=0.01,
                lower_bound=0.000001,
                upper_bound=10.0,
            ),
        ],
    }

    initial_condition_specs_by_model = {
        "irreversible": make_initial_condition_specs(),
        "reversible": make_initial_condition_specs(),
    }

    result = fit_global_observable_model_comparison(
        models=models,
        dataset=dataset,
        parameter_specs_by_model=parameter_specs_by_model,
        initial_condition_specs_by_model=initial_condition_specs_by_model,
        observed_species_by_model="A",
        settings_by_model=FitSettings(
            species_mapping={},
            rtol=1e-8,
            atol=1e-10,
        ),
        scale_initial_guess=1.0,
        scale_lower_bound=0.0,
        scale_upper_bound=5.0,
        offset_initial_guess=0.0,
        offset_lower_bound=-1.0,
        offset_upper_bound=1.0,
    )

    written_files = export_global_observable_model_comparison(
        result=result,
        output_dir=tmp_path,
    )

    assert "global_observable_model_comparison" in written_files
    assert "global_observable_model_comparison_failures" in written_files

    assert (tmp_path / "global_observable_model_comparison.csv").exists()
    assert (
        tmp_path / "global_observable_model_comparison_failures.csv"
    ).exists()

    comparison_table = pd.read_csv(
        tmp_path / "global_observable_model_comparison.csv"
    )

    assert set(comparison_table["model"]) == {"irreversible", "reversible"}
