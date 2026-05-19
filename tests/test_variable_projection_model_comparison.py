import numpy as np
import pandas as pd
import pytest

from odefit.data.dataset import Dataset
from odefit.fitting.fit_settings import FitSettings
from odefit.fitting.initial_condition_spec import InitialConditionSpec
from odefit.fitting.parameter_spec import ParameterSpec
from odefit.fitting.variable_projection_model_comparison import (
    build_variable_projection_model_specs_from_texts,
    export_variable_projection_model_comparison,
    fit_global_observable_variable_projection_model_comparison,
)
from odefit.model.model_spec import build_model_spec


def make_hsqc_like_dataset(
    n_peaks: int = 8,
    n_timepoints: int = 30,
    true_k: float = 0.4,
) -> Dataset:
    timepoints = np.linspace(0.0, 8.0, n_timepoints)
    a_values = np.exp(-true_k * timepoints)

    data = {
        "time": timepoints,
    }

    for index in range(n_peaks):
        scale = 1.0 + 0.1 * index
        offset = 0.05 * index
        data[f"P{index + 1}"] = scale * a_values + offset

    dataframe = pd.DataFrame(data)

    signal_columns = [
        column
        for column in dataframe.columns
        if column != "time"
    ]

    return Dataset(
        raw_dataframe=dataframe,
        time_column="time",
        signal_columns=signal_columns,
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
    return build_variable_projection_model_specs_from_texts(
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


def test_build_variable_projection_model_specs_from_texts():
    models = build_variable_projection_model_specs_from_texts(
        {
            "irreversible": "A>B",
            "reversible": "A-B",
        }
    )

    assert set(models) == {"irreversible", "reversible"}
    assert models["irreversible"].name == "irreversible"
    assert models["reversible"].name == "reversible"


def test_fit_global_observable_variable_projection_model_comparison():
    result = fit_global_observable_variable_projection_model_comparison(
        models=make_models(),
        dataset=make_hsqc_like_dataset(),
        parameter_specs_by_model=make_parameter_specs_by_model(),
        initial_condition_specs_by_model=make_initial_condition_specs_by_model(),
        observed_species_by_model="A",
        settings_by_model=FitSettings(
            species_mapping={},
            method="trf",
            loss="linear",
            rtol=1e-8,
            atol=1e-10,
        ),
        fit_scale=True,
        fit_offset=True,
        backend="numpy",
        method="LSODA",
        sort_by="aic",
    )

    assert set(result.fit_results) == {"irreversible", "reversible"}
    assert result.failures == []

    assert result.best_model_name in {"irreversible", "reversible"}
    assert result.best_result.success

    assert "rank" in result.comparison_table.columns
    assert "model" in result.comparison_table.columns
    assert "aic" in result.comparison_table.columns
    assert "parameter_k1f" in result.comparison_table.columns
    assert len(result.comparison_table) == 2


def test_fit_global_observable_variable_projection_model_comparison_records_failure():
    models = {
        "valid": build_model_spec("A>B"),
        "invalid_observed_species": build_model_spec("A>B"),
    }

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

    result = fit_global_observable_variable_projection_model_comparison(
        models=models,
        dataset=make_hsqc_like_dataset(),
        parameter_specs_by_model=parameter_specs_by_model,
        initial_condition_specs_by_model=initial_condition_specs_by_model,
        observed_species_by_model={
            "valid": "A",
            "invalid_observed_species": "missing",
        },
        settings_by_model=FitSettings(
            species_mapping={},
            method="trf",
            loss="linear",
            rtol=1e-8,
            atol=1e-10,
        ),
        fit_scale=True,
        fit_offset=True,
        backend="numpy",
        method="LSODA",
    )

    assert set(result.fit_results) == {"valid"}
    assert len(result.failures) == 1
    assert result.failures[0].model_name == "invalid_observed_species"
    assert result.best_result.success


def test_fit_global_observable_variable_projection_model_comparison_raises_if_all_fail():
    models = {
        "bad": build_model_spec("A>B"),
    }

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

    with pytest.raises(RuntimeError, match="All variable projection model comparison fits failed"):
        fit_global_observable_variable_projection_model_comparison(
            models=models,
            dataset=make_hsqc_like_dataset(),
            parameter_specs_by_model=parameter_specs_by_model,
            initial_condition_specs_by_model=initial_condition_specs_by_model,
            observed_species_by_model="missing",
            settings_by_model=FitSettings(
                species_mapping={},
            ),
        )


def test_export_variable_projection_model_comparison(tmp_path):
    result = fit_global_observable_variable_projection_model_comparison(
        models=make_models(),
        dataset=make_hsqc_like_dataset(
            n_peaks=4,
            n_timepoints=20,
        ),
        parameter_specs_by_model=make_parameter_specs_by_model(),
        initial_condition_specs_by_model=make_initial_condition_specs_by_model(),
        observed_species_by_model="A",
        settings_by_model=FitSettings(
            species_mapping={},
            method="trf",
            loss="linear",
            rtol=1e-8,
            atol=1e-10,
        ),
        fit_scale=True,
        fit_offset=True,
        backend="numpy",
        method="LSODA",
    )

    written_files = export_variable_projection_model_comparison(
        result=result,
        output_dir=tmp_path,
        export_best_fit=True,
    )

    assert "variable_projection_model_comparison" in written_files
    assert "variable_projection_model_comparison_failures" in written_files
    assert "best_fit_projected_observables" in written_files
    assert "best_fit_projected_fitted_parameters" in written_files

    assert (tmp_path / "variable_projection_model_comparison.csv").exists()
    assert (tmp_path / "variable_projection_model_comparison_failures.csv").exists()

    best_fit_dir = tmp_path / "best_fit"

    assert (best_fit_dir / "projected_observables.csv").exists()
    assert (best_fit_dir / "projected_predictions.csv").exists()
    assert (best_fit_dir / "projected_residuals.csv").exists()
    assert (best_fit_dir / "projected_simulation.csv").exists()
    assert (best_fit_dir / "projected_fit_statistics.csv").exists()
    assert (best_fit_dir / "projected_fitted_parameters.csv").exists()

    comparison = pd.read_csv(tmp_path / "variable_projection_model_comparison.csv")

    assert len(comparison) == 2
    assert "aic" in comparison.columns
