import numpy as np
import pytest

from odefit.engines.numba_projection import is_numba_available
from odefit.engines.registry import (
    available_engine_names,
    describe_available_engines,
    get_engine_bundle,
)


def test_numba_projection_engine_is_registered():
    assert "numba_projection" in available_engine_names()


def test_describe_available_engines_includes_numba_projection():
    descriptions = describe_available_engines()

    names = {entry["name"] for entry in descriptions}

    assert "numba_projection" in names


@pytest.mark.skipif(
    not is_numba_available(),
    reason="numba is not installed",
)
def test_numba_projection_engine_single_species_matches_reference():
    reference = get_engine_bundle("reference")
    numba_engine = get_engine_bundle("numba_projection")

    species_values = np.linspace(0.0, 1.0, 20)
    observed_values = 2.5 * species_values - 0.3

    reference_result = reference.projection.project_single_species(
        observed_values=observed_values,
        species_values=species_values,
        fit_scale=True,
        fit_offset=True,
    )

    numba_result = numba_engine.projection.project_single_species(
        observed_values=observed_values,
        species_values=species_values,
        fit_scale=True,
        fit_offset=True,
    )

    assert np.isclose(numba_result.scale, reference_result.scale)
    assert np.isclose(numba_result.offset, reference_result.offset)
    assert np.allclose(numba_result.predicted, reference_result.predicted)
    assert np.allclose(numba_result.residuals, reference_result.residuals)
    assert np.isclose(numba_result.rss, reference_result.rss)


@pytest.mark.skipif(
    not is_numba_available(),
    reason="numba is not installed",
)
def test_numba_projection_engine_supports_scale_only():
    numba_engine = get_engine_bundle("numba_projection")

    species_values = np.linspace(0.1, 1.0, 20)
    observed_values = 3.0 * species_values

    result = numba_engine.projection.project_single_species(
        observed_values=observed_values,
        species_values=species_values,
        fit_scale=True,
        fit_offset=False,
    )

    assert np.isclose(result.scale, 3.0)
    assert np.isclose(result.offset, 0.0)
    assert result.rss < 1e-20


@pytest.mark.skipif(
    not is_numba_available(),
    reason="numba is not installed",
)
def test_numba_projection_engine_multispecies_delegates_to_reference():
    numba_engine = get_engine_bundle("numba_projection")

    x1 = np.linspace(0.0, 1.0, 20)
    x2 = 1.0 - x1

    species_matrix = np.column_stack([x1, x2])
    observed_values = 1.5 * x1 - 0.2 * x2 + 0.1

    result = numba_engine.projection.project_multispecies(
        observed_values=observed_values,
        species_matrix=species_matrix,
        species_names=["A", "B"],
        fit_offset=True,
    )

    assert "A" in result.coefficients
    assert "B" in result.coefficients
    assert result.rss < 1e-20
