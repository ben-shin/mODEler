import numpy as np

from odefit.engines.registry import get_engine_bundle


def test_reference_projection_engine_has_batched_single_species_method():
    bundle = get_engine_bundle("reference")

    assert hasattr(bundle.projection, "project_single_species_batch")


def test_reference_batched_single_species_projection_matches_single_column_loop():
    bundle = get_engine_bundle("reference")

    time = np.linspace(0.0, 5.0, 12)
    species_values = np.exp(-0.4 * time)

    observed_matrix = np.column_stack(
        [
            1.5 * species_values + 0.1,
            0.7 * species_values - 0.2,
            2.1 * species_values + 0.0,
        ]
    )

    batch_result = bundle.projection.project_single_species_batch(
        observed_matrix=observed_matrix,
        species_values=species_values,
        fit_scale=True,
        fit_offset=True,
    )

    assert batch_result.scales.shape == (3,)
    assert batch_result.offsets.shape == (3,)
    assert batch_result.predicted.shape == observed_matrix.shape
    assert batch_result.residuals.shape == observed_matrix.shape
    assert batch_result.rss_by_observable.shape == (3,)

    for column_index in range(observed_matrix.shape[1]):
        single_result = bundle.projection.project_single_species(
            observed_values=observed_matrix[:, column_index],
            species_values=species_values,
            fit_scale=True,
            fit_offset=True,
        )

        assert np.isclose(
            batch_result.scales[column_index],
            single_result.scale,
        )
        assert np.isclose(
            batch_result.offsets[column_index],
            single_result.offset,
        )
        assert np.allclose(
            batch_result.predicted[:, column_index],
            single_result.predicted,
        )
        assert np.allclose(
            batch_result.residuals[:, column_index],
            single_result.residuals,
        )
        assert np.isclose(
            batch_result.rss_by_observable[column_index],
            single_result.rss,
        )
