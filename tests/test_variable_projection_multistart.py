import numpy as np
import pandas as pd
import pytest

from odefit.data.dataset import Dataset
from odefit.fitting.fit_settings import FitSettings
from odefit.fitting.initial_condition_spec import InitialConditionSpec
from odefit.fitting.parameter_spec import ParameterSpec
from odefit.fitting.variable_projection_multistart import (
    build_variable_projection_multistart_parameter_sets,
    export_variable_projection_multistart_summary,
    fit_global_observable_variable_projection_multistart,
)
from odefit.model.model_spec import build_model_spec


def make_hsqc_like_dataset(
    n_peaks: int = 10,
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


def make_parameter_specs() -> list[ParameterSpec]:
    return [
        ParameterSpec(
            name="k1f",
            initial_guess=0.1,
            lower_bound=0.001,
            upper_bound=10.0,
        )
    ]


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


def test_build_variable_projection_multistart_parameter_sets():
    parameter_specs = make_parameter_specs()

    parameter_sets = build_variable_projection_multistart_parameter_sets(
        parameter_specs=parameter_specs,
        n_starts=4,
        random_seed=1,
        log_uniform=True,
    )

    assert len(parameter_sets) == 4

    assert parameter_sets[0][0].initial_guess == pytest.approx(0.1)

    randomized_guesses = [
        parameter_set[0].initial_guess
        for parameter_set in parameter_sets[1:]
    ]

    assert all(0.001 <= guess <= 10.0 for guess in randomized_guesses)
    assert len(set(randomized_guesses)) > 1


def test_build_variable_projection_multistart_parameter_sets_rejects_zero_starts():
    with pytest.raises(ValueError, match="n_starts"):
        build_variable_projection_multistart_parameter_sets(
            parameter_specs=make_parameter_specs(),
            n_starts=0,
        )


def test_fit_global_observable_variable_projection_multistart():
    model = build_model_spec("A>B")
    dataset = make_hsqc_like_dataset(
        n_peaks=8,
        n_timepoints=30,
        true_k=0.4,
    )

    result = fit_global_observable_variable_projection_multistart(
        model=model,
        dataset=dataset,
        parameter_specs=make_parameter_specs(),
        initial_condition_specs=make_initial_condition_specs(),
        observed_species="A",
        settings=FitSettings(
            species_mapping={},
            method="trf",
            loss="linear",
            rtol=1e-8,
            atol=1e-10,
        ),
        n_starts=4,
        random_seed=1,
        sort_by="aic",
        show_progress=False,
    )

    assert result.n_submitted == 4
    assert result.n_successful == 4
    assert result.n_failed == 0
    assert result.failures == []

    assert result.best_result.success
    assert result.best_result.fitted_parameters["k1f"] == pytest.approx(
        0.4,
        rel=1e-3,
    )

    assert result.best_index in {0, 1, 2, 3}
    assert len(result.all_results) == 4
    assert len(result.starting_parameter_sets) == 4

    assert "rank" in result.comparison_table.columns
    assert "start_index" in result.comparison_table.columns
    assert "aic" in result.comparison_table.columns
    assert "parameter_k1f" in result.comparison_table.columns

    assert list(result.comparison_table["rank"]) == [1, 2, 3, 4]


def test_fit_global_observable_variable_projection_multistart_records_failure():
    model = build_model_spec("A>B")
    dataset = make_hsqc_like_dataset()

    bad_initial_conditions = make_initial_condition_specs()
    bad_initial_conditions[0].fixed = False

    with pytest.raises(RuntimeError, match="All variable projection multistart fits failed"):
        fit_global_observable_variable_projection_multistart(
            model=model,
            dataset=dataset,
            parameter_specs=make_parameter_specs(),
            initial_condition_specs=bad_initial_conditions,
            observed_species="A",
            settings=FitSettings(
                species_mapping={},
            ),
            n_starts=2,
            random_seed=1,
            show_progress=False,
        )


def test_export_variable_projection_multistart_summary(tmp_path):
    model = build_model_spec("A>B")
    dataset = make_hsqc_like_dataset(
        n_peaks=4,
        n_timepoints=20,
        true_k=0.4,
    )

    result = fit_global_observable_variable_projection_multistart(
        model=model,
        dataset=dataset,
        parameter_specs=make_parameter_specs(),
        initial_condition_specs=make_initial_condition_specs(),
        observed_species="A",
        settings=FitSettings(
            species_mapping={},
            method="trf",
            loss="linear",
            rtol=1e-8,
            atol=1e-10,
        ),
        n_starts=3,
        random_seed=1,
        show_progress=False,
    )

    written_files = export_variable_projection_multistart_summary(
        result=result,
        output_dir=tmp_path,
        export_best_fit=True,
    )

    assert "variable_projection_multistart_comparison" in written_files
    assert "variable_projection_multistart_starting_parameters" in written_files
    assert "variable_projection_multistart_failures" in written_files
    assert "best_fit_projected_observables" in written_files
    assert "best_fit_projected_fitted_parameters" in written_files

    assert (tmp_path / "variable_projection_multistart_comparison.csv").exists()
    assert (
        tmp_path / "variable_projection_multistart_starting_parameters.csv"
    ).exists()
    assert (tmp_path / "variable_projection_multistart_failures.csv").exists()

    best_fit_dir = tmp_path / "best_fit"

    assert (best_fit_dir / "projected_observables.csv").exists()
    assert (best_fit_dir / "projected_predictions.csv").exists()
    assert (best_fit_dir / "projected_residuals.csv").exists()
    assert (best_fit_dir / "projected_simulation.csv").exists()
    assert (best_fit_dir / "projected_fit_statistics.csv").exists()
    assert (best_fit_dir / "projected_fitted_parameters.csv").exists()

    comparison = pd.read_csv(tmp_path / "variable_projection_multistart_comparison.csv")

    assert len(comparison) == 3
    assert "aic" in comparison.columns
