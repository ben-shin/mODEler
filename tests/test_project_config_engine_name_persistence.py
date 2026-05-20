import json

from odefit.api.project_config import (
    collect_engine_names,
    ensure_engine_name,
    load_project_payload,
    normalize_project_payload,
    save_project_payload,
    validate_project_engines,
)


def test_ensure_engine_name_preserves_existing_value():
    config = {
        "engine_name": "numba_projection",
        "data": "data.csv",
    }

    normalized = ensure_engine_name(config)

    assert normalized["engine_name"] == "numba_projection"


def test_ensure_engine_name_adds_default_when_missing():
    config = {
        "data": "data.csv",
    }

    normalized = ensure_engine_name(config)

    assert normalized["engine_name"] == "reference"


def test_normalize_raw_fit_config_preserves_engine_name():
    payload = {
        "engine_name": "numba_projection",
        "data": "data.csv",
        "time_column": "time",
        "model_text": "A -> B",
    }

    normalized = normalize_project_payload(payload)

    assert normalized["engine_name"] == "numba_projection"
    assert normalized["project_schema_version"] == 1


def test_normalize_nested_fit_config_preserves_engine_name():
    payload = {
        "project_name": "example",
        "fit_config": {
            "engine_name": "jax_projection",
            "data": "data.csv",
            "model_text": "A -> B",
        },
    }

    normalized = normalize_project_payload(payload)

    assert normalized["fit_config"]["engine_name"] == "jax_projection"


def test_normalize_workflow_configs_preserves_each_engine_name():
    payload = {
        "project_name": "example",
        "workflow_configs": {
            "main_fit": {
                "engine_name": "numba_projection",
                "data": "data.csv",
            },
            "bootstrap": {
                "engine_name": "reference",
                "data": "data.csv",
            },
            "profile_likelihood": {
                "data": "data.csv",
            },
        },
    }

    normalized = normalize_project_payload(payload)

    assert normalized["workflow_configs"]["main_fit"]["engine_name"] == "numba_projection"
    assert normalized["workflow_configs"]["bootstrap"]["engine_name"] == "reference"
    assert normalized["workflow_configs"]["profile_likelihood"]["engine_name"] == "reference"


def test_save_load_project_payload_roundtrip_preserves_engine_names(tmp_path):
    path = tmp_path / "project.json"

    payload = {
        "project_name": "fcs_project",
        "fit_config": {
            "engine_name": "numba_projection",
            "data": "fcs.csv",
            "time_column": "time_min",
        },
        "workflow_configs": {
            "surface_fit": {
                "engine_name": "numba_projection",
                "data": "surface.csv",
            },
            "quick_reference_check": {
                "engine_name": "reference",
                "data": "surface.csv",
            },
        },
    }

    save_project_payload(payload, path)

    loaded = load_project_payload(path)

    assert loaded["fit_config"]["engine_name"] == "numba_projection"
    assert loaded["workflow_configs"]["surface_fit"]["engine_name"] == "numba_projection"
    assert loaded["workflow_configs"]["quick_reference_check"]["engine_name"] == "reference"

    raw = json.loads(path.read_text())

    assert raw["fit_config"]["engine_name"] == "numba_projection"


def test_collect_engine_names_returns_paths():
    payload = {
        "fit_config": {
            "engine_name": "numba_projection",
        },
        "workflow_configs": {
            "jax_run": {
                "engine_name": "jax_projection",
            }
        },
    }

    engine_names = collect_engine_names(payload)

    assert engine_names["/fit_config"] == "numba_projection"
    assert engine_names["/workflow_configs/jax_run"] == "jax_projection"


def test_validate_project_engines_reports_invalid_engine():
    payload = {
        "fit_config": {
            "engine_name": "not_a_real_engine",
        }
    }

    validation = validate_project_engines(payload)

    assert validation["valid"] is False
    assert validation["validations"]["/fit_config"]["valid"] is False
    assert validation["validations"]["/fit_config"]["available"] is False
