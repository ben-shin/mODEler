import subprocess
import sys

import pandas as pd


def test_triage_engine_benchmarks_script_runs(tmp_path):
    summary_path = tmp_path / "engine_benchmark_summary.csv"

    dataframe = pd.DataFrame(
        [
            {
                "benchmark": "single_species_projection",
                "engine_name": "reference",
                "available": True,
                "n_timepoints": 50,
                "n_observables": None,
                "median_seconds": 0.01,
                "mean_seconds": 0.011,
                "speedup_vs_reference_median": 1.0,
            },
            {
                "benchmark": "single_species_projection",
                "engine_name": "numba_projection",
                "available": True,
                "n_timepoints": 50,
                "n_observables": None,
                "median_seconds": 0.004,
                "mean_seconds": 0.005,
                "speedup_vs_reference_median": 2.5,
            },
            {
                "benchmark": "single_species_variable_projection_fit",
                "engine_name": "reference",
                "available": True,
                "n_timepoints": 30,
                "n_observables": 100,
                "median_seconds": 0.5,
                "mean_seconds": 0.55,
                "speedup_vs_reference_median": 1.0,
            },
            {
                "benchmark": "single_species_variable_projection_fit",
                "engine_name": "numba_projection",
                "available": True,
                "n_timepoints": 30,
                "n_observables": 100,
                "median_seconds": 0.48,
                "mean_seconds": 0.52,
                "speedup_vs_reference_median": 1.04,
            },
        ]
    )

    dataframe.to_csv(summary_path, index=False)

    output_dir = tmp_path / "triage"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/triage_engine_benchmarks.py",
            "--summary-csv",
            str(summary_path),
            "--output-dir",
            str(output_dir),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr

    markdown_path = output_dir / "optimization_triage.md"
    json_path = output_dir / "optimization_triage.json"
    csv_path = output_dir / "optimization_triage.csv"

    assert markdown_path.exists()
    assert json_path.exists()
    assert csv_path.exists()

    markdown = markdown_path.read_text()

    assert "Engine Optimization Triage" in markdown
    assert "Kernel speedup does not translate" in markdown


def test_triage_engine_benchmarks_handles_missing_summary(tmp_path):
    output_dir = tmp_path / "triage"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/triage_engine_benchmarks.py",
            "--summary-csv",
            str(tmp_path / "missing.csv"),
            "--output-dir",
            str(output_dir),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr

    markdown_path = output_dir / "optimization_triage.md"

    assert markdown_path.exists()
    assert "No benchmark summary data found" in markdown_path.read_text()
