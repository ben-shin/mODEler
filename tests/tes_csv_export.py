import numpy as np
import pandas as pd

from odefit.data.dataset import Dataset
from odefit.export.csv_export import (
    build_simulated_curves_table,
    export_fit_result_tables,
    write_dataframe_csv,
)
from odefit.fitting.fit_result import FitResult
from odefit.fitting.initial_condition_spec import InitialConditionSpec
from odefit.fitting.parameter_spec import ParameterSpec
from odefit.simulation.simulation_result import SimulationResult


def make_simulation_result() -> SimulationResult:
    return SimulationResult(
        timepoints=np.array([0.0, 1.0, 2.0]),
        species=["A", "B"],
        values=np.array(
            [
                [1.0, 0.0],
                [0.6, 0.4],
                [0.3, 0.7],
            ]
        ),
    )


def make_fit_result() -> FitResult:
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
        residuals=np.array([0.0, 0.1, -0.1]),
        statistics={
            "rss": 0.02,
            "rmse": 0.08165,
            "aic": -10.0,
            "bic": -9.0,
            "n_residuals": 3.0,
            "n_parameters": 1.0,
        },
        simulation_result=simulation_result,
        nfev=8,
        cost=0.01,
        fitted_initial_conditions={
            "A": 1.0,
            "B": 0.0,
        },
        initial_conditions={
            "A": 1.0,
            "B": 0.0,
        },
    )


def test_write_dataframe_csv(tmp_path):
    dataframe = pd.DataFrame(
        {
            "A": [1, 2],
            "B": [3, 4],
        }
    )

    output_path = tmp_path / "table.csv"

    written_path = write_dataframe_csv(
        dataframe=dataframe,
        file_path=output_path,
    )

    assert written_path == output_path
    assert output_path.exists()

    loaded = pd.read_csv(output_path)

    assert list(loaded.columns) == ["A", "B"]
    assert list(loaded["A"]) == [1, 2]
    assert list(loaded["B"]) == [3, 4]


def test_build_simulated_curves_table():
    simulation_result = make_simulation_result()

    table = build_simulated_curves_table(simulation_result)

    assert list(table.columns) == ["time", "A", "B"]
    assert list(table["time"]) == [0.0, 1.0, 2.0]
    assert list(table["A"]) == [1.0, 0.6, 0.3]
    assert list(table["B"]) == [0.0, 0.4, 0.7]


def test_export_fit_result_tables_writes_core_files(tmp_path):
    fit_result = make_fit_result()

    written_files = export_fit_result_tables(
        fit_result=fit_result,
        output_dir=tmp_path,
    )

    assert "fit_statistics" in written_files
    assert "simulated_curves" in written_files

    assert (tmp_path / "fit_statistics.csv").exists()
    assert (tmp_path / "simulated_curves.csv").exists()

    statistics = pd.read_csv(tmp_path / "fit_statistics.csv")
    simulated_curves = pd.read_csv(tmp_path / "simulated_curves.csv")

    assert list(statistics.columns) == ["statistic", "value"]
    assert list(simulated_curves.columns) == ["time", "A", "B"]


def test_export_fit_result_tables_writes_parameter_table(tmp_path):
    fit_result = make_fit_result()

    parameter_specs = [
        ParameterSpec(
            name="k1f",
            initial_guess=0.1,
            lower_bound=0.0,
            upper_bound=10.0,
        )
    ]

    written_files = export_fit_result_tables(
        fit_result=fit_result,
        output_dir=tmp_path,
        parameter_specs=parameter_specs,
    )

    assert "fitted_parameters" in written_files
    assert (tmp_path / "fitted_parameters.csv").exists()

    table = pd.read_csv(tmp_path / "fitted_parameters.csv")

    assert list(table["parameter"]) == ["k1f"]
    assert list(table["initial_guess"]) == [0.1]
    assert list(table["fitted_value"]) == [0.5]


def test_export_fit_result_tables_writes_initial_condition_table(tmp_path):
    fit_result = make_fit_result()

    initial_condition_specs = [
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

    written_files = export_fit_result_tables(
        fit_result=fit_result,
        output_dir=tmp_path,
        initial_condition_specs=initial_condition_specs,
    )

    assert "fitted_initial_conditions" in written_files
    assert (tmp_path / "fitted_initial_conditions.csv").exists()

    table = pd.read_csv(tmp_path / "fitted_initial_conditions.csv")

    assert list(table["species"]) == ["A", "B"]
    assert list(table["initial_guess"]) == [1.0, 0.0]
    assert list(table["fitted_value"]) == [1.0, 0.0]


def test_export_fit_result_tables_writes_residual_table(tmp_path):
    fit_result = make_fit_result()

    dataset = Dataset(
        raw_dataframe=pd.DataFrame(
            {
                "time": [0.0, 1.0, 2.0],
                "A": [1.0, 0.5, 0.25],
                "B": [0.0, 0.5, 0.75],
            }
        ),
        time_column="time",
        signal_columns=["A", "B"],
    )

    written_files = export_fit_result_tables(
        fit_result=fit_result,
        output_dir=tmp_path,
        dataset=dataset,
        species_mapping={
            "A": "A",
            "B": "B",
        },
    )

    assert "residuals" in written_files
    assert (tmp_path / "residuals.csv").exists()

    table = pd.read_csv(tmp_path / "residuals.csv")

    assert list(table.columns) == [
        "time",
        "A_observed",
        "A_fit",
        "A_residual",
        "B_observed",
        "B_fit",
        "B_residual",
    ]
