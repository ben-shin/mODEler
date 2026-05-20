import subprocess
import sys
from pathlib import Path


def test_projection_engine_batch_benchmark_script_runs(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "scripts/benchmark_projection_engine_batch_methods.py",
            "--engines",
            "reference",
            "numba_projection",
            "--n-timepoints",
            "8",
            "--n-observables",
            "6",
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

    json_path = tmp_path / "projection_engine_batch_method_benchmarks.json"
    csv_path = tmp_path / "projection_engine_batch_method_benchmarks.csv"

    assert json_path.exists()
    assert csv_path.exists()


def test_projection_engine_batch_benchmark_handles_missing_jax(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "scripts/benchmark_projection_engine_batch_methods.py",
            "--engines",
            "jax_projection",
            "--n-timepoints",
            "8",
            "--n-observables",
            "6",
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

    json_path = tmp_path / "projection_engine_batch_method_benchmarks.json"

    assert json_path.exists()

    text = json_path.read_text()

    assert "jax_projection" in text
