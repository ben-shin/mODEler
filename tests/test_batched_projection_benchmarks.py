import numpy as np
import pytest

from odefit.benchmarking.batched_projection import (
    JAX_AVAILABLE,
    assert_projection_outputs_close,
    export_batched_projection_benchmarks,
    jax_vectorized_batched_single_species_projection,
    make_batched_single_species_projection_data,
    numpy_loop_batched_single_species_projection,
    numpy_vectorized_batched_single_species_projection,
    run_batched_projection_benchmarks,
)


def test_make_batched_single_species_projection_data_shapes():
    species_values, observed_matrix = make_batched_single_species_projection_data(
        n_timepoints=7,
        n_observables=5,
    )

    assert species_values.shape == (7,)
    assert observed_matrix.shape == (7, 5)


def test_numpy_vectorized_matches_loop_projection():
    species_values, observed_matrix = make_batched_single_species_projection_data(
        n_timepoints=12,
        n_observables=8,
        noise=0.0,
    )

    loop = numpy_loop_batched_single_species_projection(
        species_values=species_values,
        observed_matrix=observed_matrix,
    )

    vectorized = numpy_vectorized_batched_single_species_projection(
        species_values=species_values,
        observed_matrix=observed_matrix,
    )

    assert_projection_outputs_close(
        vectorized,
        loop,
        rtol=1e-9,
        atol=1e-10,
    )


@pytest.mark.skipif(
    not JAX_AVAILABLE,
    reason="JAX is not installed",
)
def test_jax_vectorized_matches_numpy_vectorized_projection():
    species_values, observed_matrix = make_batched_single_species_projection_data(
        n_timepoints=12,
        n_observables=8,
        noise=0.0,
    )

    numpy_result = numpy_vectorized_batched_single_species_projection(
        species_values=species_values,
        observed_matrix=observed_matrix,
    )

    jax_result = jax_vectorized_batched_single_species_projection(
        species_values=species_values,
        observed_matrix=observed_matrix,
    )

    assert_projection_outputs_close(
        jax_result,
        numpy_result,
        rtol=1e-7,
        atol=1e-9,
    )


def test_run_batched_projection_benchmarks_fast():
    results = run_batched_projection_benchmarks(
        n_timepoints=8,
        n_observables=6,
        n_repeats=1,
        include_loop=True,
        include_jax=False,
    )

    names = {result.name for result in results}

    assert "numpy_loop" in names
    assert "numpy_vectorized" in names

    for result in results:
        assert result.n_repeats == 1
        assert result.median_seconds >= 0.0


def test_export_batched_projection_benchmarks(tmp_path):
    results = run_batched_projection_benchmarks(
        n_timepoints=8,
        n_observables=6,
        n_repeats=1,
        include_loop=False,
        include_jax=False,
    )

    written_files = export_batched_projection_benchmarks(
        results=results,
        output_dir=tmp_path,
    )

    assert written_files["json"].exists()
    assert written_files["csv"].exists()
