import numpy as np
import pytest

from odefit.engines.numba_projection import is_numba_available
from odefit.engines.registry import get_engine_bundle


@pytest.mark.skipif(
    not is_numba_available(),
    reason="Numba is not installed",
)
def test_numba_batched_projection_matches_reference_scale_offset():
    reference = get_engine_bundle("reference")
    numba_engine = get_engine_bundle("numba_projection")

    time = np.linspace(0.0, 5.0, 25)
    species_values = np.exp(-0.4 * time)

    observed_matrix = np.column_stack(
        [
            1.5 * species_values + 0.1,
            0.7 * species_values - 0.2,
            2.1 * species_values + 0.0,
            -0.3 * species_values + 1.2,
        ]
    )

    reference_result = reference.projection.project_single_species_batch(
        observed_matrix=observed_matrix,
        species_values=species_values,
        fit_scale=True,
        fit_offset=True,
    )

    numba_result = numba_engine.projection.project_single_species_batch(
        observed_matrix=observed_matrix,
        species_values=species_values,
        fit_scale=True,
        fit_offset=True,
    )

    np.testing.assert_allclose(
        numba_result.scales,
        reference_result.scales,
        rtol=1e-8,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        numba_result.offsets,
        reference_result.offsets,
        rtol=1e-8,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        numba_result.predicted,
        reference_result.predicted,
        rtol=1e-8,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        numba_result.residuals,
        reference_result.residuals,
        rtol=1e-8,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        numba_result.rss_by_observable,
        reference_result.rss_by_observable,
        rtol=1e-8,
        atol=1e-10,
    )
    assert np.isclose(
        numba_result.rss,
        reference_result.rss,
        rtol=1e-8,
        atol=1e-10,
    )


@pytest.mark.skipif(
    not is_numba_available(),
    reason="Numba is not installed",
)
def test_numba_batched_projection_matches_reference_scale_only():
    reference = get_engine_bundle("reference")
    numba_engine = get_engine_bundle("numba_projection")

    time = np.linspace(0.0, 5.0, 25)
    species_values = np.exp(-0.4 * time)

    observed_matrix = np.column_stack(
        [
            1.5 * species_values,
            0.7 * species_values,
            2.1 * species_values,
        ]
    )

    reference_result = reference.projection.project_single_species_batch(
        observed_matrix=observed_matrix,
        species_values=species_values,
        fit_scale=True,
        fit_offset=False,
    )

    numba_result = numba_engine.projection.project_single_species_batch(
        observed_matrix=observed_matrix,
        species_values=species_values,
        fit_scale=True,
        fit_offset=False,
    )

    np.testing.assert_allclose(
        numba_result.scales,
        reference_result.scales,
        rtol=1e-8,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        numba_result.offsets,
        reference_result.offsets,
        rtol=1e-8,
        atol=1e-10,
    )
    assert np.isclose(
        numba_result.rss,
        reference_result.rss,
        rtol=1e-8,
        atol=1e-10,
    )


@pytest.mark.skipif(
    not is_numba_available(),
    reason="Numba is not installed",
)
def test_numba_batched_projection_matches_reference_offset_only():
    reference = get_engine_bundle("reference")
    numba_engine = get_engine_bundle("numba_projection")

    time = np.linspace(0.0, 5.0, 25)
    species_values = np.exp(-0.4 * time)

    observed_matrix = np.column_stack(
        [
            species_values + 0.1,
            species_values - 0.2,
            species_values + 1.2,
        ]
    )

    reference_result = reference.projection.project_single_species_batch(
        observed_matrix=observed_matrix,
        species_values=species_values,
        fit_scale=False,
        fit_offset=True,
    )

    numba_result = numba_engine.projection.project_single_species_batch(
        observed_matrix=observed_matrix,
        species_values=species_values,
        fit_scale=False,
        fit_offset=True,
    )

    np.testing.assert_allclose(
        numba_result.scales,
        reference_result.scales,
        rtol=1e-8,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        numba_result.offsets,
        reference_result.offsets,
        rtol=1e-8,
        atol=1e-10,
    )
    assert np.isclose(
        numba_result.rss,
        reference_result.rss,
        rtol=1e-8,
        atol=1e-10,
    )


@pytest.mark.skipif(
    not is_numba_available(),
    reason="Numba is not installed",
)
def test_numba_batched_projection_missing_values_fallback_matches_reference():
    reference = get_engine_bundle("reference")
    numba_engine = get_engine_bundle("numba_projection")

    time = np.linspace(0.0, 5.0, 25)
    species_values = np.exp(-0.4 * time)

    observed_matrix = np.column_stack(
        [
            1.5 * species_values + 0.1,
            0.7 * species_values - 0.2,
        ]
    )

    observed_matrix[3, 0] = np.nan

    reference_result = reference.projection.project_single_species_batch(
        observed_matrix=observed_matrix,
        species_values=species_values,
        fit_scale=True,
        fit_offset=True,
    )

    numba_result = numba_engine.projection.project_single_species_batch(
        observed_matrix=observed_matrix,
        species_values=species_values,
        fit_scale=True,
        fit_offset=True,
    )

    np.testing.assert_allclose(
        numba_result.scales,
        reference_result.scales,
        rtol=1e-8,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        numba_result.offsets,
        reference_result.offsets,
        rtol=1e-8,
        atol=1e-10,
    )
    assert np.isclose(
        numba_result.rss,
        reference_result.rss,
        rtol=1e-8,
        atol=1e-10,
    )
