import subprocess
import sys


def test_fcs_progress_demo_script_runs():
    result = subprocess.run(
        [
            sys.executable,
            "examples/backend_api/demo_fcs_progress_callbacks.py",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Model comparison table" in result.stdout
    assert "fcs_model_comparison" in result.stdout
