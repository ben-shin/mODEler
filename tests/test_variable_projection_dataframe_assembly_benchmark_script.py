import subprocess
import sys


def test_variable_projection_dataframe_assembly_benchmark_script_runs(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "scripts/benchmark_variable_projection_dataframe_assembly.py",
            "--n-timepoints",
            "10",
            "--n-observables",
            "8",
            "--n-repeats",
            "1",
            "--output-dir",
            str(tmp_path),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr

    json_path = tmp_path / "variable_projection_dataframe_assembly.json"
    csv_path = tmp_path / "variable_projection_dataframe_assembly.csv"

    assert json_path.exists()
    assert csv_path.exists()

    assert "warnings=0" in result.stdout
