import numpy as np
import pandas as pd

from odefit.data.dataset import Dataset
from odefit.export.bundle_export import (
    export_dataset_files,
    export_fit_bundle,
    export_fit_plots,
    export_model_files,
    write_text_file,
)
from odefit.fitting.fit_result import FitResult
from odefit.fitting.initial_condition_spec import InitialConditionSpec
from odefit.fitting.parameter_spec import ParameterSpec
from odefit.model.model_spec import build_model_spec
from odefit.simulation.simulation_result import SimulationResult


def make_model():
    return build_model_spec("A>B")


def make_dataset(normalized: bool = False):
    raw_dataframe = pd.DataFrame(
        {
            "time": [0.0, 1.0, 2.0],
            "A": [1.0, 0.5, 0.25],
            "B": [0.0, 0.5, 0.75],
        }
    )

    normalized_dataframe = None

    if normalized:
        normalized_dataframe = pd.DataFrame(
            {
                "time": [0.0, 1.0, 2.0],
                "A": [1.0, 0.5, 0.25],
                "B": [0.0, 0.5, 0.75],
            }
        )

    return Dataset(
        raw_dataframe=raw_dataframe,
        normalized_dataframe=normalized_dataframe,
        normalization_method="manual" if normalized else None,
        time_column="time",
        signal_columns=["A", "B"],
    )


def make_simulation_result():
    return SimulationResult(
        timepoints=np.array([0.0, 1.0, 2.0]),
        species=["A", "B"],
        values=np.array(
            [
                [1.0, 0.0],
                [0.5, 0.5],
                [0.25, 0.75],
            ]
        ),
    )


def make_fit_result():
    simulation_result = make_simulation_result()

    return FitResult(
        success=True,
        message="Converged",
        fitted_parameters={
            "k1f": 0.5,
        },
        initial_parameters={
            "k1f": 0.1,
        },
        residuals=np.array([0.0, 0.0, 0.0]),
        statistics={
            "rss": 0.0,
            "rmse": 0.0,
            "aic": -100.0,
            "bic": -99.0,
            "n_residuals": 6.0,
            "n_parameters": 1.0,
        },
        simulation_result=simulation_result,
        nfev=5,
        cost=0.0,
        fitted_initial_conditions={
            "A": 1.0,
            "B": 0.0,
        },
        initial_conditions={
            "A": 1.0,
            "B": 0.0,
        },
    )


def make_parameter_specs():
    return [
        ParameterSpec(
            name="k1f",
            initial_guess=0.1,
            lower_bound=0.0,
            upper_bound=10.0,
        )
    ]


def make_initial_condition_specs():
    return [
        InitialConditionSpec(
            species="A",
            initial_guess=1.0,
            lower_bound=0.0,
            upper_bound=5.0,
            fixed=False,
        ),
        InitialConditionSpec(
            species="B",
            initial_guess=0.0,
            lower_bound=0.0,
            upper_bound=5.0,
            fixed=True,
            fixed_value=0.0,
        ),
    ]


def test_write_text_file(tmp_path):
    output_path = tmp_path / "example.txt"

    written_path = write_text_file(
        text="hello",
        file_path=output_path,
    )

    assert written_path == output_path
    assert output_path.exists()
    assert output_path.read_text() == "hello"


def test_export_model_files(tmp_path):
    model = make_model()

    written_files = export_model_files(
        model=model,
        output_dir=tmp_path,
    )

    assert "model_definition" in written_files
    assert "generated_odes" in written_files

    assert (tmp_path / "model_definition.txt").exists()
    assert (tmp_path / "generated_odes.txt").exists()

    assert (tmp_path / "model_definition.txt").read_text() == "A>B"

    generated_odes = (tmp_path / "generated_odes.txt").read_text()

    assert "dA/dt" in generated_odes
    assert "dB/dt" in generated_odes


def test_export_dataset_files_raw_only(tmp_path):
    dataset = make_dataset(normalized=False)

    written_files = export_dataset_files(
        dataset=dataset,
        output_dir=tmp_path,
    )

    assert "raw_data" in written_files
    assert "normalized_data" not in written_files

    assert (tmp_path / "raw_data.csv").exists()
    assert not (tmp_path / "normalized_data.csv").exists()


def test_export_dataset_files_with_normalized_data(tmp_path):
    dataset = make_dataset(normalized=True)

    written_files = export_dataset_files(
        dataset=dataset,
        output_dir=tmp_path,
    )

    assert "raw_data" in written_files
    assert "normalized_data" in written_files

    assert (tmp_path / "raw_data.csv").exists()
    assert (tmp_path / "normalized_data.csv").exists()


def test_export_fit_plots(tmp_path):
    fit_result = make_fit_result()
    dataset = make_dataset()

    written_files = export_fit_plots(
        fit_result=fit_result,
        dataset=dataset,
        species_mapping={
            "A": "A",
            "B": "B",
        },
        output_dir=tmp_path,
    )

    assert "observed_vs_fitted_plot" in written_files
    assert "residuals_plot" in written_files

    assert (tmp_path / "observed_vs_fitted.png").exists()
    assert (tmp_path / "residuals.png").exists()


def test_export_fit_bundle_writes_expected_files(tmp_path):
    model = make_model()
    dataset = make_dataset()
    fit_result = make_fit_result()

    written_files = export_fit_bundle(
        fit_result=fit_result,
        model=model,
        dataset=dataset,
        output_dir=tmp_path,
        parameter_specs=make_parameter_specs(),
        initial_condition_specs=make_initial_condition_specs(),
        species_mapping={
            "A": "A",
            "B": "B",
        },
        include_plots=True,
    )

    expected_keys = {
        "model_definition",
        "generated_odes",
        "raw_data",
        "fit_statistics",
        "simulated_curves",
        "fitted_parameters",
        "fitted_initial_conditions",
        "residuals",
        "observed_vs_fitted_plot",
        "residuals_plot",
    }

    assert expected_keys.issubset(set(written_files.keys()))

    expected_files = [
        "model_definition.txt",
        "generated_odes.txt",
        "raw_data.csv",
        "fit_statistics.csv",
        "simulated_curves.csv",
        "fitted_parameters.csv",
        "fitted_initial_conditions.csv",
        "residuals.csv",
        "observed_vs_fitted.png",
        "residuals.png",
    ]

    for filename in expected_files:
        assert (tmp_path / filename).exists()


def test_export_fit_bundle_can_skip_plots(tmp_path):
    model = make_model()
    dataset = make_dataset()
    fit_result = make_fit_result()

    written_files = export_fit_bundle(
        fit_result=fit_result,
        model=model,
        dataset=dataset,
        output_dir=tmp_path,
        parameter_specs=make_parameter_specs(),
        initial_condition_specs=make_initial_condition_specs(),
        species_mapping={
            "A": "A",
            "B": "B",
        },
        include_plots=False,
    )

    assert "observed_vs_fitted_plot" not in written_files
    assert "residuals_plot" not in written_files

    assert not (tmp_path / "observed_vs_fitted.png").exists()
    assert not (tmp_path / "residuals.png").exists()
