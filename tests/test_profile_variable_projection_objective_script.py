import subprocess
import sys


def test_profile_variable_projection_objective_script_runs(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "scripts/profile_variable_projection_objective.py",
            "--engines",
            "reference",
            "numba_projection",
            "--n-timepoints",
            "10",
            "--n-observables",
            "5",
            "--n-repeats",
            "2",
            "--output-dir",
            str(tmp_path),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr

    json_path = tmp_path / "variable_projection_objective_profile.json"
    csv_path = tmp_path / "variable_projection_objective_profile.csv"

    assert json_path.exists()
    assert csv_path.exists()

    text = json_path.read_text()

    assert "reference" in text
    assert "numba_projection" in text
    assert "solve_median_seconds" in text


def test_profile_variable_projection_objective_handles_jax(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "scripts/profile_variable_projection_objective.py",
            "--engines",
            "jax_projection",
            "--n-timepoints",
            "10",
            "--n-observables",
            "5",
            "--n-repeats",
            "2",
            "--output-dir",
            str(tmp_path),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr

    json_path = tmp_path / "variable_projection_objective_profile.json"

    assert json_path.exists()
    assert "jax_projection" in json_path.read_text()
