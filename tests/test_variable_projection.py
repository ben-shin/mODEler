import numpy as np
import pandas as pd
import pytest

from odefit.data.dataset import Dataset
from odefit.fitting.fit_settings import FitSettings
from odefit.fitting.initial_condition_spec import InitialConditionSpec
from odefit.fitting.parameter_spec import ParameterSpec
from odefit.fitting.variable_projection import (
    export_variable_projection_fit,
    fit_global_observable_model_variable_projection,
    project_observables_onto_species,
    solve_scale_offset,
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


def test_solve_scale_offset_exact():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = 2.5 * x + 0.7

    scale, offset = solve_scale_offset(
        x_values=x,
        y_values=y,
        fit_scale=True,
        fit_offset=True,
    )

    assert scale == pytest.approx(2.5)
    assert offset == pytest.approx(0.7)


def test_solve_scale_offset_scale_only():
    x = np.array([1.0, 2.0, 3.0])
    y = 4.0 * x

    scale, offset = solve_scale_offset(
        x_values=x,
        y_values=y,
        fit_scale=True,
        fit_offset=False,
    )

    assert scale == pytest.approx(4.0)
    assert offset == pytest.approx(0.0)


def test_project_observables_onto_species_exact():
    timepoints = np.array([0.0, 1.0, 2.0])
    x = np.array([1.0, 0.5, 0.25])

    dataframe = pd.DataFrame(
        {
            "time": timepoints,
            "P1": 2.0 * x + 0.1,
            "P2": 3.0 * x + 0.2,
        }
    )

    result = project_observables_onto_species(
        timepoints=timepoints,
        simulated_species_values=x,
        observed_dataframe=dataframe,
        signal_columns=["P1", "P2"],
        fit_scale=True,
        fit_offset=True,
    )

    assert result.rss == pytest.approx(0.0, abs=1e-12)
    assert result.n_observations == 6
    assert result.n_linear_parameters == 4

    p1 = result.observable_table.loc[
        result.observable_table["data_column"] == "P1"
    ].iloc[0]

    assert p1["scale"] == pytest.approx(2.0)
    assert p1["offset"] == pytest.approx(0.1)


def test_variable_projection_global_fit_recovers_first_order_rate():
    model = build_model_spec("A>B")
    dataset = make_hsqc_like_dataset(
        n_peaks=10,
        n_timepoints=30,
        true_k=0.4,
    )

    result = fit_global_observable_model_variable_projection(
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
        backend="numpy",
        method="LSODA",
    )

    assert result.success
    assert result.fitted_parameters["k1f"] == pytest.approx(0.4, rel=1e-3)
    assert result.statistics["rss"] == pytest.approx(0.0, abs=1e-8)
    assert result.statistics["n_nonlinear_parameters"] == 1
    assert result.statistics["n_linear_parameters"] == 20
    assert len(result.observable_table) == 10


def test_variable_projection_rejects_nonfixed_initial_conditions():
    model = build_model_spec("A>B")
    dataset = make_hsqc_like_dataset()

    initial_condition_specs = make_initial_condition_specs()
    initial_condition_specs[0].fixed = False

    with pytest.raises(NotImplementedError, match="fixed initial conditions"):
        fit_global_observable_model_variable_projection(
            model=model,
            dataset=dataset,
            parameter_specs=make_parameter_specs(),
            initial_condition_specs=initial_condition_specs,
            observed_species="A",
        )


def test_export_variable_projection_fit(tmp_path):
    model = build_model_spec("A>B")
    dataset = make_hsqc_like_dataset(
        n_peaks=3,
        n_timepoints=20,
        true_k=0.4,
    )

    result = fit_global_observable_model_variable_projection(
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
    )

    written_files = export_variable_projection_fit(
        result=result,
        output_dir=str(tmp_path),
    )

    assert "projected_observables" in written_files
    assert "projected_predictions" in written_files
    assert "projected_residuals" in written_files
    assert "projected_simulation" in written_files
    assert "projected_fit_statistics" in written_files
    assert "projected_fitted_parameters" in written_files

    assert (tmp_path / "projected_observables.csv").exists()
    assert (tmp_path / "projected_predictions.csv").exists()
    assert (tmp_path / "projected_residuals.csv").exists()
    assert (tmp_path / "projected_simulation.csv").exists()
    assert (tmp_path / "projected_fit_statistics.csv").exists()
    assert (tmp_path / "projected_fitted_parameters.csv").exists()
