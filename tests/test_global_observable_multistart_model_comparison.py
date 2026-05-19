import numpy as np
import pandas as pd
import pytest

from odefit.data.dataset import Dataset
from odefit.fitting.fit_settings import FitSettings
from odefit.fitting.global_observable_multistart_model_comparison import (
    export_global_observable_multistart_model_comparison,
    fit_global_observable_multistart_model_comparison,
)
from odefit.fitting.global_observable_model_comparison import (
    build_model_specs_from_texts,
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


def make_models():
    return build_model_specs_from_texts(
        {
            "irreversible": "A>B",
            "reversible": "A-B",
        }
    )


def make_parameter_specs_by_model():
    return {
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


def make_initial_condition_specs_by_model():
    return {
        "irreversible": make_initial_condition_specs(),
        "reversible": make_initial_condition_specs(),
    }


def test_fit_global_observable_multistart_model_comparison():
    result = fit_global_observable_multistart_model_comparison(
        models=make_models(),
        dataset=make_global_observable_dataset(),
        parameter_specs_by_model=make_parameter_specs_by_model(),
        initial_condition_specs_by_model=make_initial_condition_specs_by_model(),
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
        n_starts=2,
        n_workers=1,
        random_seed=1,
        sort_by="aic",
    )

    assert set(result.multistart_results) == {"irreversible", "reversible"}
    assert set(result.best_fit_results_by_model) == {"irreversible", "reversible"}
    assert result.failures == []

    assert result.best_model_name in {"irreversible", "reversible"}
    assert result.best_fit_result.success

    assert "model" in result.comparison_table.columns
    assert "rank" in result.comparison_table.columns
    assert len(result.comparison_table) == 2

    for multistart_result in result.multistart_results.values():
        assert multistart_result.n_submitted == 2
        assert multistart_result.n_successful == 2
        assert multistart_result.best_result.success


def test_fit_global_observable_multistart_model_comparison_records_failure():
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

    result = fit_global_observable_multistart_model_comparison(
        models=models,
        dataset=make_global_observable_dataset(),
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
        n_starts=2,
        n_workers=1,
        random_seed=1,
    )

    assert set(result.multistart_results) == {"valid"}
    assert len(result.failures) == 1
    assert result.failures[0].model_name == "invalid_observed_species"
    assert result.best_fit_result.success


def test_fit_global_observable_multistart_model_comparison_raises_if_all_fail():
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
        fit_global_observable_multistart_model_comparison(
            models=models,
            dataset=make_global_observable_dataset(),
            parameter_specs_by_model=parameter_specs_by_model,
            initial_condition_specs_by_model=initial_condition_specs_by_model,
            observed_species_by_model="missing",
            settings_by_model=FitSettings(
                species_mapping={},
            ),
            n_starts=2,
            n_workers=1,
        )


def test_export_global_observable_multistart_model_comparison(tmp_path):
    result = fit_global_observable_multistart_model_comparison(
        models=make_models(),
        dataset=make_global_observable_dataset(),
        parameter_specs_by_model=make_parameter_specs_by_model(),
        initial_condition_specs_by_model=make_initial_condition_specs_by_model(),
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
        n_starts=2,
        n_workers=1,
        random_seed=1,
    )

    written_files = export_global_observable_multistart_model_comparison(
        result=result,
        output_dir=tmp_path,
    )

    assert "global_observable_multistart_model_comparison" in written_files
    assert "global_observable_multistart_model_comparison_failures" in written_files
    assert "global_observable_multistart_model_summary" in written_files

    assert (
        tmp_path / "global_observable_multistart_model_comparison.csv"
    ).exists()
    assert (
        tmp_path / "global_observable_multistart_model_comparison_failures.csv"
    ).exists()
    assert (
        tmp_path / "global_observable_multistart_model_summary.csv"
    ).exists()

    assert (
        tmp_path
        / "per_model_multistart"
        / "irreversible"
        / "global_observable_multistart_comparison.csv"
    ).exists()

    assert (
        tmp_path
        / "per_model_multistart"
        / "reversible"
        / "global_observable_multistart_comparison.csv"
    ).exists()

    comparison_table = pd.read_csv(
        tmp_path / "global_observable_multistart_model_comparison.csv"
    )

    assert set(comparison_table["model"]) == {"irreversible", "reversible"}
