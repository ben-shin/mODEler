import subprocess
import sys
from pathlib import Path


def test_backend_api_demo_script_runs():
    script = Path("examples/backend_api/demo_backend_api_serialization.py")

    result = subprocess.run(
        [
            sys.executable,
            str(script),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr

    output_path = Path("examples/backend_api/backend_api_fit_payload.json")

    assert output_path.exists()
    assert output_path.stat().st_size > 0
