import subprocess
import sys
from pathlib import Path


def test_project_io_demo_script_runs():
    script = Path("examples/backend_api/demo_project_io.py")

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

    project_path = Path("examples/backend_api/example_project.modeler.json")
    summary_path = Path("examples/backend_api/example_project_summary.json")

    assert project_path.exists()
    assert project_path.stat().st_size > 0

    assert summary_path.exists()
    assert summary_path.stat().st_size > 0
