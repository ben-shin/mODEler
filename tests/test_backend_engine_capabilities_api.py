import json
import subprocess
import sys

from odefit.api.backend import (
    get_backend_engine_capabilities,
    validate_backend_engine_name,
)


def test_get_backend_engine_capabilities_returns_gui_payload():
    payload = get_backend_engine_capabilities()

    assert payload["default_engine"] == "reference"
    assert "available_engine_names" in payload
    assert "engines" in payload

    assert "reference" in payload["available_engine_names"]

    engines = payload["engines"]

    assert isinstance(engines, list)
    assert engines

    by_name = {
        engine["name"]: engine
        for engine in engines
    }

    assert "reference" in by_name

    reference = by_name["reference"]

    assert reference["available"] is True
    assert reference["is_default"] is True
    assert "capabilities" in reference

    # The exact component names come from BackendEngineBundle.capabilities().
    assert isinstance(reference["capabilities"], dict)
    assert reference["capabilities"]


def test_get_backend_engine_capabilities_is_json_serializable():
    payload = get_backend_engine_capabilities()

    encoded = json.dumps(payload)
    decoded = json.loads(encoded)

    assert decoded["default_engine"] == "reference"


def test_validate_backend_engine_name_accepts_reference():
    payload = validate_backend_engine_name("reference")

    assert payload["valid"] is True
    assert payload["available"] is True
    assert payload["resolved_name"] == "reference"
    assert "capabilities" in payload


def test_validate_backend_engine_name_rejects_unknown_engine():
    payload = validate_backend_engine_name("not_a_real_engine")

    assert payload["valid"] is False
    assert payload["available"] is False
    assert payload["error_type"] == "ValueError"
    assert "Unknown engine bundle" in payload["error_message"]


def test_show_backend_engines_for_gui_script_runs(tmp_path):
    output_path = tmp_path / "backend_engines.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/show_backend_engines_for_gui.py",
            "--output",
            str(output_path),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr

    assert output_path.exists()

    payload = json.loads(output_path.read_text())

    assert payload["default_engine"] == "reference"
    assert "reference" in payload["available_engine_names"]
