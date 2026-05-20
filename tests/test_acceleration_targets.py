from odefit.benchmarking.acceleration_targets import (
    benchmark_multispecies_projection_kernel,
    benchmark_simulation_solve,
    benchmark_single_species_projection_kernel,
    export_benchmark_results,
    make_synthetic_global_observable_dataset,
    make_synthetic_multispecies_dataset,
    run_acceleration_target_benchmarks,
)


def test_make_synthetic_global_observable_dataset():
    dataset = make_synthetic_global_observable_dataset(
        n_timepoints=5,
        n_peaks=3,
    )

    assert dataset.time_column == "time"
    assert len(dataset.signal_columns) == 3
    assert dataset.raw_dataframe.shape == (5, 4)


def test_make_synthetic_multispecies_dataset():
    dataset = make_synthetic_multispecies_dataset(
        n_timepoints=5,
        n_peaks=3,
    )

    assert dataset.time_column == "time"
    assert len(dataset.signal_columns) == 3
    assert dataset.raw_dataframe.shape == (5, 4)


def test_benchmark_simulation_solve_runs():
    result = benchmark_simulation_solve(
        n_timepoints=5,
        n_repeats=1,
    )

    assert result.name == "simulation_solve"
    assert result.n_repeats == 1
    assert len(result.times_seconds) == 1
    assert result.median_seconds >= 0.0


def test_projection_kernel_benchmarks_run():
    single = benchmark_single_species_projection_kernel(
        n_timepoints=5,
        n_repeats=2,
    )

    multi = benchmark_multispecies_projection_kernel(
        n_timepoints=5,
        n_species=2,
        n_repeats=2,
    )

    assert single.name == "single_species_projection_kernel"
    assert multi.name == "multispecies_projection_kernel"

    assert len(single.times_seconds) == 2
    assert len(multi.times_seconds) == 2


def test_run_acceleration_target_benchmarks_fast(tmp_path):
    results = run_acceleration_target_benchmarks(
        work_dir=tmp_path,
        n_repeats_fast=1,
        n_repeats_fit=1,
        include_slow=False,
    )

    names = {result.name for result in results}

    assert "simulation_solve" in names
    assert "single_species_projection_kernel" in names
    assert "multispecies_projection_kernel" in names
    assert "variable_projection_fit" in names
    assert "multispecies_variable_projection_fit" in names


def test_export_benchmark_results(tmp_path):
    results = [
        benchmark_simulation_solve(
            n_timepoints=5,
            n_repeats=1,
        )
    ]

    written_files = export_benchmark_results(
        results=results,
        output_dir=tmp_path,
    )

    assert written_files["json"].exists()
    assert written_files["csv"].exists()
