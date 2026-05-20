import json

from odefit.api.project_io import (
    PROJECT_SCHEMA_VERSION,
    attach_result_payload,
    create_project,
    load_project,
    project_to_config,
    save_project,
    summarize_project,
    update_project_config,
    validate_project_dict,
)


def test_create_project_has_required_fields():
    project = create_project(
        name="Example project",
        workflow="fit",
        config={
            "model_text": "A -> B",
            "observed_species": "A",
        },
    )

    assert project.schema_version == PROJECT_SCHEMA_VERSION
    assert project.name == "Example project"
    assert project.workflow == "fit"
    assert project.config["model_text"] == "A -> B"
    assert project.result_payload is None


def test_save_and_load_project_roundtrip(tmp_path):
    project = create_project(
        name="Roundtrip project",
        workflow="fit",
        config={
            "model_text": "A -> B",
            "observed_species": "A",
        },
        notes=["Initial GUI alpha project."],
        tags=["hsqc", "fit"],
    )

    path = tmp_path / "project.modeler.json"

    save_project(
        project=project,
        file_path=path,
    )

    loaded = load_project(path)

    assert loaded.project_id == project.project_id
    assert loaded.name == "Roundtrip project"
    assert loaded.workflow == "fit"
    assert loaded.config["model_text"] == "A -> B"
    assert loaded.notes == ["Initial GUI alpha project."]
    assert loaded.tags == ["hsqc", "fit"]


def test_project_json_is_valid_json(tmp_path):
    project = create_project(
        name="JSON project",
        workflow="fit",
        config={
            "model_text": "A -> B",
            "observed_species": "A",
        },
    )

    path = tmp_path / "project.modeler.json"
    save_project(project, path)

    with path.open("r") as handle:
        data = json.load(handle)

    validate_project_dict(data)

    assert data["schema_version"] == PROJECT_SCHEMA_VERSION


def test_attach_result_payload():
    project = create_project(
        name="Payload project",
        workflow="fit",
        config={
            "model_text": "A -> B",
            "observed_species": "A",
        },
    )

    payload = {
        "workflow": "fit",
        "result": {
            "success": True,
            "fitted_parameters": {
                "k1f": 0.4,
            },
        },
    }

    attach_result_payload(
        project=project,
        result_payload=payload,
    )

    assert project.result_payload == payload


def test_update_project_config():
    project = create_project(
        name="Update project",
        workflow="fit",
        config={
            "model_text": "A -> B",
        },
    )

    update_project_config(
        project=project,
        config_updates={
            "observed_species": "A",
            "fit_offset": True,
        },
    )

    assert project.config["observed_species"] == "A"
    assert project.config["fit_offset"] is True


def test_project_to_config_returns_config_copy():
    project = create_project(
        name="Config project",
        workflow="fit",
        config={
            "model_text": "A -> B",
            "observed_species": "A",
        },
    )

    config = project_to_config(project)

    assert config == project.config
    assert config is not project.config


def test_summarize_project():
    project = create_project(
        name="Summary project",
        workflow="bootstrap",
        config={
            "model_text": "A -> B",
        },
        notes=["note 1"],
        tags=["hsqc"],
    )

    summary = summarize_project(project)

    assert summary["name"] == "Summary project"
    assert summary["workflow"] == "bootstrap"
    assert summary["has_result_payload"] is False
    assert summary["n_notes"] == 1
    assert summary["tags"] == ["hsqc"]


def test_validate_project_dict_rejects_missing_required_key():
    bad_project = {
        "schema_version": PROJECT_SCHEMA_VERSION,
        "name": "Bad project",
        "workflow": "fit",
        "config": {},
    }

    try:
        validate_project_dict(bad_project)
    except ValueError as exc:
        assert "missing required keys" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
