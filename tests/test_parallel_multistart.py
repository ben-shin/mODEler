import numpy as np
import pandas as pd
import pytest

from odefit.data.dataset import Dataset
from odefit.fitting.fit_settings import FitSettings
from odefit.fitting.initial_condition_spec import InitialConditionSpec
from odefit.fitting.parallel_multistart import (
    build_multistart_parameter_spec_sets,
    export_parallel_multistart_summary,
    fit_multistart_parallel,
)
from odefit.fitting.parameter_spec import ParameterSpec
from odefit.model.model_spec import build_model_spec


def make_dataset() -> Dataset:
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

    return Dataset(
        raw_dataframe=dataframe,
        time_column="time",
        signal_columns=["A", "B"],
    )


def make_initial_condition_specs() -> list[InitialConditionSpec]:
    return [
        InitialConditionSpec(
            species="A",
            initial_guess=1.0,
            lower_bound=0.0,
            upper_bound=10.0,
            fixed=True,
            fixed_value=1.0,
        ),
        InitialConditionSpec(
            species="B",
            initial_guess=0.0,
            lower_bound=0.0,
            upper_bound=10.0,
            fixed=True,
            fixed_value=0.0,
        ),
    ]


def make_parameter_specs() -> list[ParameterSpec]:
    return [
        ParameterSpec(
            name="k1f",
            initial_guess=0.1,
            lower_bound=0.001,
            upper_bound=2.0,
        )
    ]


def make_settings() -> FitSettings:
    return FitSettings(
        species_mapping={
            "A": "A",
            "B": "B",
        },
        rtol=1e-8,
        atol=1e-10,
    )


def test_build_multistart_parameter_spec_sets():
    parameter_specs = make_parameter_specs()

    spec_sets = build_multistart_parameter_spec_sets(
        parameter_specs=parameter_specs,
        n_starts=4,
        random_seed=1,
    )

    assert len(spec_sets) == 4

    # First start preserves original guess.
    assert spec_sets[0][0].initial_guess == 0.1

    # Later starts are sampled inside bounds.
    for spec_set in spec_sets[1:]:
        sampled_guess = spec_set[0].initial_guess
        assert 0.001 <= sampled_guess <= 2.0


def test_build_multistart_parameter_spec_sets_rejects_zero_starts():
    with pytest.raises(ValueError):
        build_multistart_parameter_spec_sets(
            parameter_specs=make_parameter_specs(),
            n_starts=0,
        )


def test_fit_multistart_parallel_returns_successful_result():
    model = build_model_spec("A>B")
    dataset = make_dataset()

    result = fit_multistart_parallel(
        model=model,
        dataset=dataset,
        parameter_specs=make_parameter_specs(),
        initial_condition_specs=make_initial_condition_specs(),
        settings=make_settings(),
        n_starts=4,
        n_workers=2,
        random_seed=1,
    )

    assert result.n_submitted == 4
    assert result.n_successful == 4
    assert result.n_failed == 0

    assert result.successful_result.best_result.success
    assert result.successful_result.best_index in {0, 1, 2, 3}

    assert result.successful_result.best_result.fitted_parameters[
        "k1f"
    ] == pytest.approx(0.5, rel=1e-2)

    assert "rank" in result.successful_result.comparison_table.columns
    assert "model" in result.successful_result.comparison_table.columns
    assert "aic" in result.successful_result.comparison_table.columns


def test_fit_multistart_parallel_can_collect_failures():
    model = build_model_spec("A>B")
    dataset = make_dataset()

    bad_parameter_specs = [
        ParameterSpec(
            name="k1f",
            initial_guess=0.1,
            lower_bound=0.001,
            upper_bound=float("inf"),
        )
    ]

    with pytest.raises(ValueError):
        build_multistart_parameter_spec_sets(
            parameter_specs=bad_parameter_specs,
            n_starts=2,
        )


def test_export_parallel_multistart_summary(tmp_path):
    model = build_model_spec("A>B")
    dataset = make_dataset()

    result = fit_multistart_parallel(
        model=model,
        dataset=dataset,
        parameter_specs=make_parameter_specs(),
        initial_condition_specs=make_initial_condition_specs(),
        settings=make_settings(),
        n_starts=2,
        n_workers=2,
        random_seed=1,
    )

    written_files = export_parallel_multistart_summary(
        parallel_result=result,
        output_dir=tmp_path,
    )

    assert "parallel_multistart_comparison" in written_files
    assert "parallel_multistart_failures" in written_files

    assert (tmp_path / "parallel_multistart_comparison.csv").exists()
    assert (tmp_path / "parallel_multistart_failures.csv").exists()

    comparison = pd.read_csv(tmp_path / "parallel_multistart_comparison.csv")

    assert "rank" in comparison.columns
    assert "model" in comparison.columns
