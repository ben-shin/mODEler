import numpy as np
import pandas as pd
import pytest

from odefit.data.dataset import Dataset
from odefit.fitting.fit_settings import FitSettings
from odefit.fitting.initial_condition_spec import InitialConditionSpec
from odefit.fitting.multistart import (
    export_multistart_comparison,
    fit_multistart,
    parameter_specs_to_initial_guess_dict,
    sample_initial_guess,
    sample_parameter_specs,
)
from odefit.fitting.parameter_spec import ParameterSpec
from odefit.model.model_spec import build_model_spec


def make_dataset() -> Dataset:
    true_k = 0.5
    timepoints = np.linspace(0.0, 5.0, 30)

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


def test_sample_initial_guess_linear_with_zero_lower_bound():
    rng = np.random.default_rng(1)

    value = sample_initial_guess(
        lower_bound=0.0,
        upper_bound=10.0,
        rng=rng,
        log_uniform=True,
    )

    assert 0.0 <= value <= 10.0


def test_sample_initial_guess_log_uniform_positive_bounds():
    rng = np.random.default_rng(1)

    value = sample_initial_guess(
        lower_bound=0.001,
        upper_bound=10.0,
        rng=rng,
        log_uniform=True,
    )

    assert 0.001 <= value <= 10.0


def test_sample_initial_guess_rejects_infinite_bound():
    rng = np.random.default_rng(1)

    with pytest.raises(ValueError):
        sample_initial_guess(
            lower_bound=0.0,
            upper_bound=float("inf"),
            rng=rng,
        )


def test_sample_initial_guess_rejects_invalid_bounds():
    rng = np.random.default_rng(1)

    with pytest.raises(ValueError):
        sample_initial_guess(
            lower_bound=10.0,
            upper_bound=0.0,
            rng=rng,
        )


def test_parameter_specs_to_initial_guess_dict():
    parameter_specs = [
        ParameterSpec(
            name="k1f",
            initial_guess=0.1,
            lower_bound=0.0,
            upper_bound=10.0,
        ),
        ParameterSpec(
            name="k1r",
            initial_guess=0.2,
            lower_bound=0.0,
            upper_bound=10.0,
            fixed=True,
            fixed_value=0.3,
        ),
    ]

    guesses = parameter_specs_to_initial_guess_dict(parameter_specs)

    assert guesses == {
        "k1f": 0.1,
        "k1r": 0.3,
    }


def test_sample_parameter_specs_changes_free_parameter_only():
    rng = np.random.default_rng(1)

    parameter_specs = [
        ParameterSpec(
            name="k1f",
            initial_guess=0.1,
            lower_bound=0.0,
            upper_bound=10.0,
        ),
        ParameterSpec(
            name="k1r",
            initial_guess=0.2,
            lower_bound=0.0,
            upper_bound=10.0,
            fixed=True,
            fixed_value=0.2,
        ),
        ParameterSpec(
            name="k2f",
            initial_guess=0.3,
            lower_bound=0.0,
            upper_bound=10.0,
            tied_to="k1f",
        ),
    ]

    sampled = sample_parameter_specs(
        parameter_specs=parameter_specs,
        rng=rng,
    )

    by_name = {parameter.name: parameter for parameter in sampled}

    assert by_name["k1f"].initial_guess != 0.1
    assert 0.0 <= by_name["k1f"].initial_guess <= 10.0

    assert by_name["k1r"].initial_guess == 0.2
    assert by_name["k2f"].initial_guess == 0.3


def test_fit_multistart_returns_best_result():
    model = build_model_spec("A>B")
    dataset = make_dataset()

    parameter_specs = [
        ParameterSpec(
            name="k1f",
            initial_guess=0.1,
            lower_bound=0.0,
            upper_bound=2.0,
        )
    ]

    result = fit_multistart(
        model=model,
        dataset=dataset,
        parameter_specs=parameter_specs,
        initial_condition_specs=make_initial_condition_specs(),
        settings=FitSettings(
            species_mapping={
                "A": "A",
                "B": "B",
            },
            rtol=1e-8,
            atol=1e-10,
        ),
        n_starts=4,
        random_seed=1,
    )

    assert len(result.all_results) == 4
    assert len(result.starting_parameter_sets) == 4

    assert result.best_index in {0, 1, 2, 3}
    assert result.best_result.success

    assert result.best_result.fitted_parameters["k1f"] == pytest.approx(
        0.5,
        rel=1e-2,
    )

    assert "rank" in result.comparison_table.columns
    assert "model" in result.comparison_table.columns
    assert "aic" in result.comparison_table.columns


def test_fit_multistart_first_start_uses_original_guess():
    model = build_model_spec("A>B")
    dataset = make_dataset()

    parameter_specs = [
        ParameterSpec(
            name="k1f",
            initial_guess=0.123,
            lower_bound=0.0,
            upper_bound=2.0,
        )
    ]

    result = fit_multistart(
        model=model,
        dataset=dataset,
        parameter_specs=parameter_specs,
        initial_condition_specs=make_initial_condition_specs(),
        settings=FitSettings(
            species_mapping={
                "A": "A",
                "B": "B",
            },
        ),
        n_starts=2,
        random_seed=1,
    )

    assert result.starting_parameter_sets[0]["k1f"] == 0.123


def test_fit_multistart_rejects_zero_starts():
    model = build_model_spec("A>B")
    dataset = make_dataset()

    with pytest.raises(ValueError):
        fit_multistart(
            model=model,
            dataset=dataset,
            parameter_specs=[
                ParameterSpec(
                    name="k1f",
                    initial_guess=0.1,
                    lower_bound=0.0,
                    upper_bound=2.0,
                )
            ],
            initial_condition_specs=make_initial_condition_specs(),
            settings=FitSettings(
                species_mapping={
                    "A": "A",
                    "B": "B",
                },
            ),
            n_starts=0,
        )


def test_export_multistart_comparison(tmp_path):
    model = build_model_spec("A>B")
    dataset = make_dataset()

    result = fit_multistart(
        model=model,
        dataset=dataset,
        parameter_specs=[
            ParameterSpec(
                name="k1f",
                initial_guess=0.1,
                lower_bound=0.0,
                upper_bound=2.0,
            )
        ],
        initial_condition_specs=make_initial_condition_specs(),
        settings=FitSettings(
            species_mapping={
                "A": "A",
                "B": "B",
            },
        ),
        n_starts=2,
        random_seed=1,
    )

    output_path = tmp_path / "multistart_comparison.csv"

    written_path = export_multistart_comparison(
        multistart_result=result,
        file_path=output_path,
    )

    assert written_path == output_path
    assert output_path.exists()

    table = pd.read_csv(output_path)

    assert "rank" in table.columns
    assert "model" in table.columns
