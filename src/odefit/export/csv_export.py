from pathlib import Path

import pandas as pd

from odefit.data.dataset import Dataset
from odefit.fitting.fit_result import FitResult
from odefit.fitting.initial_condition_spec import InitialConditionSpec
from odefit.fitting.initial_condition_table import build_initial_condition_table
from odefit.fitting.observable_spec import ObservableSpec
from odefit.fitting.observable_table import build_observable_table
from odefit.fitting.parameter_spec import ParameterSpec
from odefit.fitting.parameter_table import build_parameter_table
from odefit.fitting.residual_table import build_residual_table
from odefit.fitting.statistics_table import build_statistics_table
from odefit.simulation.simulation_result import SimulationResult


def write_dataframe_csv(
    dataframe: pd.DataFrame,
    file_path: str | Path,
) -> Path:
    """
    Write a dataframe to CSV and return the written path.
    """

    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    dataframe.to_csv(path, index=False)

    return path


def build_simulated_curves_table(
    simulation_result: SimulationResult,
) -> pd.DataFrame:
    """
    Build a table of simulated species curves.

    Output columns:
        time, species_1, species_2, ...
    """

    data = {
        "time": simulation_result.timepoints,
    }

    for species_name in simulation_result.species:
        data[species_name] = simulation_result.get_species_values(species_name)

    return pd.DataFrame(data)


def export_fit_result_tables(
    fit_result: FitResult,
    output_dir: str | Path,
    parameter_specs: list[ParameterSpec] | None = None,
    initial_condition_specs: list[InitialConditionSpec] | None = None,
    observable_specs: list[ObservableSpec] | None = None,
    dataset: Dataset | None = None,
    species_mapping: dict[str, str] | None = None,
    use_normalized_data: bool = False,
) -> dict[str, Path]:
    """
    Export FitResult-derived tables to CSV files.

    Always exports:
        fit_statistics.csv
        simulated_curves.csv

    Optionally exports:
        fitted_parameters.csv
        fitted_initial_conditions.csv
        residuals.csv

    Returns a dictionary mapping output names to file paths.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    written_files: dict[str, Path] = {}

    statistics_table = build_statistics_table(fit_result)
    written_files["fit_statistics"] = write_dataframe_csv(
        dataframe=statistics_table,
        file_path=output_path / "fit_statistics.csv",
    )

    simulated_curves_table = build_simulated_curves_table(fit_result.simulation_result)
    written_files["simulated_curves"] = write_dataframe_csv(
        dataframe=simulated_curves_table,
        file_path=output_path / "simulated_curves.csv",
    )

    if observable_specs is not None:
        if fit_result.fitted_observables is None:
            raise ValueError("FitResult is missing fitted observables")

        observable_table = build_observable_table(
            observable_specs=observable_specs,
            fitted_observables=fit_result.fitted_observables,
        )

        written_files["fitted_observables"] = write_dataframe_csv(
            dataframe=observable_table,
            file_path=output_path / "fitted_observables.csv",
        )

    if parameter_specs is not None:
        parameter_table = build_parameter_table(
            parameter_specs=parameter_specs,
            fitted_parameters=fit_result.fitted_parameters,
        )

        written_files["fitted_parameters"] = write_dataframe_csv(
            dataframe=parameter_table,
            file_path=output_path / "fitted_parameters.csv",
        )

    if initial_condition_specs is not None:
        if fit_result.fitted_initial_conditions is None:
            raise ValueError("FitResult is missing fitted initial conditions")

        initial_condition_table = build_initial_condition_table(
            initial_condition_specs=initial_condition_specs,
            fitted_initial_conditions=fit_result.fitted_initial_conditions,
        )

        written_files["fitted_initial_conditions"] = write_dataframe_csv(
            dataframe=initial_condition_table,
            file_path=output_path / "fitted_initial_conditions.csv",
        )

    if dataset is not None or species_mapping is not None:
        if dataset is None:
            raise ValueError("dataset is required to export residuals")

        if species_mapping is None:
            raise ValueError("species_mapping is required to export residuals")

        residual_table = build_residual_table(
            dataset=dataset,
            simulation_result=fit_result.simulation_result,
            species_mapping=species_mapping,
            use_normalized_data=use_normalized_data,
        )

        written_files["residuals"] = write_dataframe_csv(
            dataframe=residual_table,
            file_path=output_path / "residuals.csv",
        )

    return written_files
