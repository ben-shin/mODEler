import json

import pandas as pd

from odefit.data.dataset import Dataset
from odefit.export.bundle_export import export_fit_bundle
from odefit.fitting.fit_settings import FitSettings
from odefit.fitting.initial_condition_spec import InitialConditionSpec
from odefit.fitting.optimizer import fit_model
from odefit.fitting.parameter_spec import ParameterSpec
from odefit.model.model_spec import build_model_spec


def test_export_fit_bundle_writes_json_metadata_files(tmp_path):
    model = build_model_spec("A>B")

    dataset = Dataset(
        raw_dataframe=pd.DataFrame(
            {
                "time": [0.0, 1.0, 2.0, 3.0],
                "A": [1.0, 0.6, 0.36, 0.216],
                "B": [0.0, 0.4, 0.64, 0.784],
            }
        ),
        time_column="time",
        signal_columns=["A", "B"],
    )

    parameter_specs = [
        ParameterSpec(
            name="k1f",
            initial_guess=0.5,
            lower_bound=0.001,
            upper_bound=10.0,
        )
    ]

    initial_condition_specs = [
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

    settings = FitSettings(
        species_mapping={
            "A": "A",
            "B": "B",
        },
        rtol=1e-8,
        atol=1e-10,
    )

    fit_result = fit_model(
        model=model,
        dataset=dataset,
        parameter_specs=parameter_specs,
        initial_condition_specs=initial_condition_specs,
        settings=settings,
    )

    written_files = export_fit_bundle(
        fit_result=fit_result,
        model=model,
        dataset=dataset,
        output_dir=tmp_path,
        parameter_specs=parameter_specs,
        initial_condition_specs=initial_condition_specs,
        species_mapping=settings.species_mapping,
        include_plots=False,
        fit_settings=settings,
        command="test-fit",
        config_path="test_config.json",
    )

    expected_files = [
        "model_metadata.json",
        "dataset_metadata.json",
        "parameter_specs.json",
        "initial_condition_specs.json",
        "observable_specs.json",
        "fit_settings.json",
        "fit_result_summary.json",
        "run_metadata.json",
    ]

    for filename in expected_files:
        assert (tmp_path / filename).exists()

    assert "model_metadata_json" in written_files
    assert "dataset_metadata_json" in written_files
    assert "fit_result_summary_json" in written_files
    assert "run_metadata_json" in written_files

    model_metadata = json.loads((tmp_path / "model_metadata.json").read_text())

    assert model_metadata["species"] == ["A", "B"]
    assert model_metadata["parameters"] == ["k1f"]

    dataset_metadata = json.loads((tmp_path / "dataset_metadata.json").read_text())

    assert dataset_metadata["time_column"] == "time"
    assert dataset_metadata["signal_columns"] == ["A", "B"]

    run_metadata = json.loads((tmp_path / "run_metadata.json").read_text())

    assert run_metadata["command"] == "test-fit"
    assert run_metadata["config_path"] == "test_config.json"

    fit_result_summary = json.loads(
        (tmp_path / "fit_result_summary.json").read_text()
    )

    assert fit_result_summary["success"] is True
    assert "k1f" in fit_result_summary["fitted_parameters"]
