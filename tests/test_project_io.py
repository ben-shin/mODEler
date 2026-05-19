import json

import pytest

from odefit.fitting.fit_settings import FitSettings
from odefit.fitting.initial_condition_spec import InitialConditionSpec
from odefit.fitting.observable_spec import ObservableSpec
from odefit.fitting.parameter_spec import ParameterSpec
from odefit.project.project_io import (
    load_project_state,
    project_state_from_dict,
    project_state_to_dict,
    save_project_state,
    validate_project_state_for_fitting,
    validate_project_state_for_simulation,
)
from odefit.project.project_state import (
    PROJECT_STATE_SCHEMA_VERSION,
    ProjectState,
    create_empty_project_state,
    create_project_state,
)


def make_project_state() -> ProjectState:
    parameter_specs = [
        ParameterSpec(
            name="k1f",
            initial_guess=0.1,
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

    observable_specs = [
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

    fit_settings = FitSettings(
        species_mapping={
            "A": "A",
            "B": "B",
        },
        method="trf",
        loss="linear",
        rtol=1e-8,
        atol=1e-10,
    )

    return create_project_state(
        project_name="test project",
        project_notes="project notes",
        model_text="A>B",
        data_path="data.csv",
        time_column="time",
        signal_columns=["A", "B"],
        normalization_method="none",
        species_mapping={
            "A": "A",
            "B": "B",
        },
        parameter_specs=parameter_specs,
        initial_condition_specs=initial_condition_specs,
        observable_specs=observable_specs,
        fit_settings=fit_settings,
        output_dir="outputs",
        last_fit_output_dir="outputs/latest",
        metadata={
            "experiment": "test",
        },
    )


def test_create_empty_project_state():
    project_state = create_empty_project_state()

    assert project_state.schema_version == PROJECT_STATE_SCHEMA_VERSION
    assert project_state.model_text == ""
    assert project_state.signal_columns == []
    assert project_state.parameter_specs == []


def test_project_state_to_dict():
    project_state = make_project_state()

    data = project_state_to_dict(project_state)

    assert data["project_name"] == "test project"
    assert data["model_text"] == "A>B"
    assert data["data_path"] == "data.csv"
    assert data["signal_columns"] == ["A", "B"]

    assert data["parameter_specs"][0]["name"] == "k1f"
    assert data["initial_condition_specs"][0]["species"] == "A"
    assert data["observable_specs"][0]["data_column"] == "A23_HN"
    assert data["fit_settings"]["method"] == "trf"


def test_project_state_from_dict_roundtrip():
    original_project_state = make_project_state()

    data = project_state_to_dict(original_project_state)

    restored_project_state = project_state_from_dict(data)

    assert restored_project_state.project_name == "test project"
    assert restored_project_state.model_text == "A>B"
    assert restored_project_state.data_path == "data.csv"
    assert restored_project_state.time_column == "time"
    assert restored_project_state.signal_columns == ["A", "B"]

    assert len(restored_project_state.parameter_specs) == 1
    assert restored_project_state.parameter_specs[0].name == "k1f"
    assert restored_project_state.parameter_specs[0].initial_guess == 0.1

    assert len(restored_project_state.initial_condition_specs) == 2
    assert restored_project_state.initial_condition_specs[0].species == "A"
    assert restored_project_state.initial_condition_specs[0].fixed is True

    assert len(restored_project_state.observable_specs) == 1
    assert restored_project_state.observable_specs[0].data_column == "A23_HN"

    assert restored_project_state.fit_settings is not None
    assert restored_project_state.fit_settings.method == "trf"
    assert restored_project_state.fit_settings.species_mapping == {
        "A": "A",
        "B": "B",
    }


def test_save_and_load_project_state(tmp_path):
    project_state = make_project_state()

    project_path = tmp_path / "project.modeler.json"

    written_path = save_project_state(
        project_state=project_state,
        file_path=project_path,
    )

    assert written_path == project_path
    assert project_path.exists()

    loaded_json = json.loads(project_path.read_text())

    assert loaded_json["project_name"] == "test project"

    loaded_project_state = load_project_state(project_path)

    assert loaded_project_state.project_name == "test project"
    assert loaded_project_state.model_text == "A>B"
    assert loaded_project_state.parameter_specs[0].name == "k1f"


def test_load_project_state_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_project_state(tmp_path / "missing.modeler.json")


def test_project_state_from_dict_rejects_newer_schema_version():
    data = project_state_to_dict(make_project_state())

    data["schema_version"] = PROJECT_STATE_SCHEMA_VERSION + 1

    with pytest.raises(ValueError):
        project_state_from_dict(data)


def test_validate_project_state_for_fitting_accepts_complete_project():
    project_state = make_project_state()

    validate_project_state_for_fitting(project_state)


def test_validate_project_state_for_fitting_rejects_incomplete_project():
    project_state = create_empty_project_state()

    with pytest.raises(ValueError, match="model_text"):
        validate_project_state_for_fitting(project_state)


def test_validate_project_state_for_simulation_accepts_complete_project():
    project_state = make_project_state()

    validate_project_state_for_simulation(project_state)


def test_validate_project_state_for_simulation_rejects_incomplete_project():
    project_state = create_empty_project_state()

    with pytest.raises(ValueError, match="model_text"):
        validate_project_state_for_simulation(project_state)
