import json
import subprocess
import sys
from pathlib import Path


EXAMPLES = [
    (
        "examples/backend_api/gui_contract_parse_fit_save.py",
        [
            "examples/backend_api/outputs/gui_contract_parsed_model.json",
            "examples/backend_api/outputs/gui_contract_fit_payload.json",
            "examples/backend_api/outputs/gui_contract_fit_project.modeler.json",
        ],
    ),
    (
        "examples/backend_api/gui_contract_model_comparison.py",
        [
            "examples/backend_api/outputs/gui_contract_model_comparison_payload.json",
        ],
    ),
    (
        "examples/backend_api/gui_contract_uncertainty.py",
        [
            "examples/backend_api/outputs/gui_contract_bootstrap_payload.json",
            "examples/backend_api/outputs/gui_contract_profile_likelihood_payload.json",
        ],
    ),
    (
        "examples/backend_api/gui_contract_multispecies_fit.py",
        [
            "examples/backend_api/outputs/gui_contract_multispecies_fit_payload.json",
        ],
    ),
]


def _run_script(script_path: str):
    result = subprocess.run(
        [
            sys.executable,
            script_path,
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def _assert_json_file(path: str):
    file_path = Path(path)

    assert file_path.exists()
    assert file_path.stat().st_size > 0

    with file_path.open("r") as handle:
        json.load(handle)


def test_gui_contract_example_scripts_run():
    for script_path, expected_outputs in EXAMPLES:
        _run_script(script_path)

        for output_path in expected_outputs:
            _assert_json_file(output_path)
