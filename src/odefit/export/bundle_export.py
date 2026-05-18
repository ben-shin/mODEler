from pathlib import Path

from odefit.data.dataset import Dataset
from odefit.export.csv_export import export_fit_result_tables, write_dataframe_csv
from odefit.export.text_export import write_generated_odes
from odefit.fitting.fit_result import FitResult
from odefit.fitting.initial_condition_spec import InitialConditionSpec
from odefit.fitting.observable_spec import ObservableSpec
from odefit.fitting.parameter_spec import ParameterSpec
from odefit.model.model_spec import ModelSpec
from odefit.plotting.observed_vs_predicted import save_observed_vs_fitted_plot
from odefit.plotting.residual_plots import save_residuals_plot


def write_text_file(
    text: str,
    file_path: str | Path,
) -> Path:
    """
    Write text to a file and return the written path.
    """

    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(text)

    return path


def export_model_files(
    model: ModelSpec,
    output_dir: str | Path,
) -> dict[str, Path]:
    """
    Export model text and generated ODEs.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    written_files: dict[str, Path] = {}

    written_files["model_definition"] = write_text_file(
        text=model.raw_text,
        file_path=output_path / "model_definition.txt",
    )

    written_files["generated_odes"] = output_path / "generated_odes.txt"

    write_generated_odes(
        model=model,
        file_path=written_files["generated_odes"],
    )

    return written_files


def export_dataset_files(
    dataset: Dataset,
    output_dir: str | Path,
) -> dict[str, Path]:
    """
    Export raw and normalized dataset CSV files.

    normalized_data.csv is only written if normalized data are available.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    written_files: dict[str, Path] = {}

    written_files["raw_data"] = write_dataframe_csv(
        dataframe=dataset.raw_dataframe,
        file_path=output_path / "raw_data.csv",
    )

    if dataset.normalized_dataframe is not None:
        written_files["normalized_data"] = write_dataframe_csv(
            dataframe=dataset.normalized_dataframe,
            file_path=output_path / "normalized_data.csv",
        )

    return written_files


def export_fit_plots(
    fit_result: FitResult,
    dataset: Dataset,
    species_mapping: dict[str, str],
    output_dir: str | Path,
    use_normalized_data: bool = False,
) -> dict[str, Path]:
    """
    Export observed-vs-fitted and residual plots.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    written_files: dict[str, Path] = {}

    written_files["observed_vs_fitted_plot"] = save_observed_vs_fitted_plot(
        dataset=dataset,
        simulation_result=fit_result.simulation_result,
        species_mapping=species_mapping,
        file_path=output_path / "observed_vs_fitted.png",
        use_normalized_data=use_normalized_data,
    )

    written_files["residuals_plot"] = save_residuals_plot(
        dataset=dataset,
        simulation_result=fit_result.simulation_result,
        species_mapping=species_mapping,
        file_path=output_path / "residuals.png",
        use_normalized_data=use_normalized_data,
    )

    return written_files


def export_fit_bundle(
    fit_result: FitResult,
    model: ModelSpec,
    dataset: Dataset,
    output_dir: str | Path,
    parameter_specs: list[ParameterSpec],
    initial_condition_specs: list[InitialConditionSpec],
    species_mapping: dict[str, str],
    use_normalized_data: bool = False,
    include_plots: bool = True,
    observable_specs: list[ObservableSpec] | None = None,
) -> dict[str, Path]:
    """
    Export a complete fit result bundle.

    The bundle contains:
    - model_definition.txt
    - generated_odes.txt
    - raw_data.csv
    - normalized_data.csv, if available
    - fitted_parameters.csv
    - fitted_initial_conditions.csv
    - fit_statistics.csv
    - residuals.csv
    - simulated_curves.csv
    - observed_vs_fitted.png, optional
    - residuals.png, optional
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    written_files: dict[str, Path] = {}

    written_files.update(
        export_model_files(
            model=model,
            output_dir=output_path,
        )
    )

    written_files.update(
        export_dataset_files(
            dataset=dataset,
            output_dir=output_path,
        )
    )

    written_files.update(
        export_fit_result_tables(
            fit_result=fit_result,
            output_dir=output_path,
            parameter_specs=parameter_specs,
            initial_condition_specs=initial_condition_specs,
            observable_specs=observable_specs,
            dataset=dataset,
            species_mapping=species_mapping,
            use_normalized_data=use_normalized_data,
        )
    )

    if include_plots:
        written_files.update(
            export_fit_plots(
                fit_result=fit_result,
                dataset=dataset,
                species_mapping=species_mapping,
                output_dir=output_path,
                use_normalized_data=use_normalized_data,
            )
        )

    return written_files
