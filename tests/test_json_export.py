import json

import numpy as np
import pandas as pd

from odefit.data.dataset import Dataset
from odefit.export.json_export import (
    build_dataset_metadata,
    build_fit_settings_metadata,
    build_initial_condition_specs_metadata,
    build_model_metadata,
    build_observable_specs_metadata,
    build_parameter_specs_metadata,
    build_run_metadata,
    export_fit_metadata_json,
    to_jsonable,
    write_json,
)
from odefit.fitting.fit_settings import FitSettings
from odefit.fitting.initial_condition_spec import InitialConditionSpec
from odefit.fitting.observable_spec import ObservableSpec
from odefit.fitting.parameter_spec import ParameterSpec
from odefit.model.model_spec import build_model_spec


def test_to_jsonable_handles_numpy_values():
    data = {
        "array": np.array([1.0, 2.0]),
        "float": np.float64(1.5),
        "int": np.int64(2),
    }

    jsonable = to_jsonable(data)

    assert jsonable == {
        "array": [1.0, 2.0],
        "float": 1.5,
        "int": 2,
    }


def test_write_json(tmp_path):
    output_path = tmp_path / "test.json"

    written_path = write_json(
        data={
            "b": 2,
            "a": 1,
        },
        file_path=output_path,
    )

    assert written_path == output_path
    assert output_path.exists()

    loaded = json.loads(output_path.read_text())

    assert loaded == {
        "a": 1,
        "b": 2,
    }


def test_build_model_metadata():
    model = build_model_spec(
        "loss: A>B",
        name="first_order_loss",
        metadata={
            "experiment": "test",
        },
    )

    metadata = build_model_metadata(model)

    assert metadata["name"] == "first_order_loss"
    assert metadata["metadata"] == {"experiment": "test"}
    assert metadata["species"] == ["A", "B"]
    assert metadata["parameters"] == ["k1f"]
    assert metadata["reactions"][0]["label"] == "loss"
    assert metadata["reactions"][0]["forward_rate"] == "k1f"


def test_build_dataset_metadata():
    dataset = Dataset(
        raw_dataframe=pd.DataFrame(
            {
                "time": [0.0, 1.0],
                "A": [1.0, 0.5],
            }
        ),
        time_column="time",
        signal_columns=["A"],
        normalization_method="none",
    )

    metadata = build_dataset_metadata(dataset)

    assert metadata["time_column"] == "time"
    assert metadata["signal_columns"] == ["A"]
    assert metadata["normalization_method"] == "none"
    assert metadata["n_rows"] == 2
    assert metadata["columns"] == ["time", "A"]


def test_build_parameter_specs_metadata():
    specs = [
        ParameterSpec(
            name="k1f",
            initial_guess=0.1,
            lower_bound=0.0,
            upper_bound=10.0,
        )
    ]

    metadata = build_parameter_specs_metadata(specs)

    assert metadata[0]["name"] == "k1f"
    assert metadata[0]["initial_guess"] == 0.1


def test_build_initial_condition_specs_metadata():
    specs = [
        InitialConditionSpec(
            species="A",
            initial_guess=1.0,
            lower_bound=0.0,
            upper_bound=2.0,
            fixed=True,
            fixed_value=1.0,
        )
    ]

    metadata = build_initial_condition_specs_metadata(specs)

    assert metadata[0]["species"] == "A"
    assert metadata[0]["fixed"] is True


def test_build_observable_specs_metadata():
    specs = [
        ObservableSpec(
            data_column="A23_HN",
            species="A",
            scale_initial_guess=1.0,
            scale_lower_bound=0.0,
            scale_upper_bound=10.0,
            offset_initial_guess=0.0,
            offset_lower_bound=-1.0,
            offset_upper_bound=1.0,
        )
    ]

    metadata = build_observable_specs_metadata(specs)

    assert metadata[0]["data_column"] == "A23_HN"
    assert metadata[0]["species"] == "A"


def test_build_fit_settings_metadata():
    settings = FitSettings(
        species_mapping={
            "A": "A",
        },
        method="trf",
        loss="linear",
        rtol=1e-6,
        atol=1e-9,
    )

    metadata = build_fit_settings_metadata(settings)

    assert metadata["species_mapping"] == {"A": "A"}
    assert metadata["method"] == "trf"
    assert metadata["loss"] == "linear"


def test_build_run_metadata():
    metadata = build_run_metadata(
        command="fit",
        config_path="config.json",
        extra_metadata={
            "note": "test",
        },
    )

    assert metadata["command"] == "fit"
    assert metadata["config_path"] == "config.json"
    assert metadata["extra_metadata"] == {"note": "test"}
    assert "created_at_utc" in metadata
    assert "python_version" in metadata
    assert "platform" in metadata


def test_export_fit_metadata_json(tmp_path):
    model = build_model_spec("A>B")

    dataset = Dataset(
        raw_dataframe=pd.DataFrame(
            {
                "time": [0.0, 1.0],
                "A": [1.0, 0.5],
            }
        ),
        time_column="time",
        signal_columns=["A"],
    )

    parameter_specs = [
        ParameterSpec(
            name="k1f",
            initial_guess=0.1,
            lower_bound=0.0,
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
        },
    )

    written_files = export_fit_metadata_json(
        output_dir=tmp_path,
        model=model,
        dataset=dataset,
        fit_settings=settings,
        parameter_specs=parameter_specs,
        initial_condition_specs=initial_condition_specs,
        command="fit",
        config_path="fit_config.json",
    )

    assert "model_metadata_json" in written_files
    assert "dataset_metadata_json" in written_files
    assert "parameter_specs_json" in written_files
    assert "initial_condition_specs_json" in written_files
    assert "observable_specs_json" in written_files
    assert "fit_settings_json" in written_files
    assert "run_metadata_json" in written_files

    assert (tmp_path / "model_metadata.json").exists()
    assert (tmp_path / "dataset_metadata.json").exists()
    assert (tmp_path / "parameter_specs.json").exists()
    assert (tmp_path / "initial_condition_specs.json").exists()
    assert (tmp_path / "observable_specs.json").exists()
    assert (tmp_path / "fit_settings.json").exists()
    assert (tmp_path / "run_metadata.json").exists()

    model_metadata = json.loads((tmp_path / "model_metadata.json").read_text())

    assert model_metadata["species"] == ["A", "B"]
    assert model_metadata["parameters"] == ["k1f"]
