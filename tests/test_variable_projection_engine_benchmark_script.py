import subprocess
import sys


def test_variable_projection_engine_benchmark_script_runs(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "scripts/benchmark_variable_projection_engines.py",
            "--engines",
            "reference",
            "numba_projection",
            "--n-timepoints",
            "10",
            "--n-observables",
            "4",
            "--n-repeats",
            "1",
            "--max-nfev",
            "20",
            "--output-dir",
            str(tmp_path),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr

    json_path = tmp_path / "variable_projection_engine_benchmarks.json"
    csv_path = tmp_path / "variable_projection_engine_benchmarks.csv"
    data_path = tmp_path / "synthetic_variable_projection_data.csv"

    assert json_path.exists()
    assert csv_path.exists()
    assert data_path.exists()

    text = json_path.read_text()

    assert "reference" in text
    assert "numba_projection" in text


def test_variable_projection_engine_benchmark_handles_missing_jax(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "scripts/benchmark_variable_projection_engines.py",
            "--engines",
            "jax_projection",
            "--n-timepoints",
            "10",
            "--n-observables",
            "4",
            "--n-repeats",
            "1",
            "--max-nfev",
            "20",
            "--output-dir",
            str(tmp_path),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr

    json_path = tmp_path / "variable_projection_engine_benchmarks.json"

    assert json_path.exists()
    assert "jax_projection" in json_path.read_text()
