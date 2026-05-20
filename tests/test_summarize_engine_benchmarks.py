import json
import subprocess
import sys


def test_summarize_engine_benchmarks_script_runs(tmp_path):
    input_path = tmp_path / "projection_engine_benchmarks.json"

    input_path.write_text(
        json.dumps(
            [
                {
                    "engine_name": "reference",
                    "available": True,
                    "benchmark": "project_single_species_batch",
                    "n_timepoints": 10,
                    "n_observables": 5,
                    "n_repeats": 2,
                    "median_seconds": 0.01,
                    "mean_seconds": 0.012,
                },
                {
                    "engine_name": "numba_projection",
                    "available": True,
                    "benchmark": "project_single_species_batch",
                    "n_timepoints": 10,
                    "n_observables": 5,
                    "n_repeats": 2,
                    "median_seconds": 0.005,
                    "mean_seconds": 0.006,
                },
                {
                    "engine_name": "jax_projection",
                    "available": False,
                    "benchmark": "project_single_species_batch",
                    "error_type": "ImportError",
                    "error_message": "JAX unavailable",
                },
            ]
        )
    )

    output_dir = tmp_path / "summary"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/summarize_engine_benchmarks.py",
            "--inputs",
            str(input_path),
            "--output-dir",
            str(output_dir),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr

    csv_path = output_dir / "engine_benchmark_summary.csv"
    markdown_path = output_dir / "engine_benchmark_summary.md"
    json_path = output_dir / "engine_benchmark_summary.json"

    assert csv_path.exists()
    assert markdown_path.exists()
    assert json_path.exists()

    markdown = markdown_path.read_text()

    assert "Engine Benchmark Summary" in markdown
    assert "numba_projection" in markdown
    assert "2" in csv_path.read_text()


def test_summarize_engine_benchmarks_handles_missing_inputs(tmp_path):
    output_dir = tmp_path / "summary"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/summarize_engine_benchmarks.py",
            "--inputs",
            str(tmp_path / "missing.json"),
            "--output-dir",
            str(output_dir),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr

    markdown_path = output_dir / "engine_benchmark_summary.md"

    assert markdown_path.exists()
    assert "No benchmark rows were found" in markdown_path.read_text()
