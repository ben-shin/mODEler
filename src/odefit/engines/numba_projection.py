from __future__ import annotations

from importlib.util import find_spec

import numpy as np

from odefit.engines.base import (
    BatchedSingleSpeciesProjectionResult,
    EngineCapabilities,
    MultispeciesProjectionResult,
    SingleSpeciesProjectionResult,
)
from odefit.engines.reference import ReferenceNumpyProjectionEngine


NUMBA_AVAILABLE = find_spec("numba") is not None

if NUMBA_AVAILABLE:
    from numba import njit
else:
    def njit(*args, **kwargs):
        def decorator(function):
            return function

        return decorator


def is_numba_available() -> bool:
    return NUMBA_AVAILABLE


@njit(cache=True)
def _fit_single_species_numba(
    x: np.ndarray,
    y: np.ndarray,
    fit_scale: bool,
    fit_offset: bool,
):
    n = 0
    sx = 0.0
    sy = 0.0
    sxx = 0.0
    sxy = 0.0

    for i in range(x.shape[0]):
        xi = x[i]
        yi = y[i]

        if np.isfinite(xi) and np.isfinite(yi):
            n += 1
            sx += xi
            sy += yi
            sxx += xi * xi
            sxy += xi * yi

    if n < 1:
        return False, 0.0, 0.0

    if fit_scale and fit_offset:
        if n < 2:
            return False, 0.0, 0.0

        denominator = n * sxx - sx * sx

        if denominator == 0.0:
            return False, 0.0, 0.0

        scale = (n * sxy - sx * sy) / denominator
        offset = (sy - scale * sx) / n
        return True, scale, offset

    if fit_scale and not fit_offset:
        if sxx == 0.0:
            return False, 0.0, 0.0

        scale = sxy / sxx
        return True, scale, 0.0

    if (not fit_scale) and fit_offset:
        offset = (sy - sx) / n
        return True, 1.0, offset

    return True, 1.0, 0.0

@njit(cache=True)
def _fit_single_species_batch_numba(
    x: np.ndarray,
    Y: np.ndarray,
    fit_scale: bool,
    fit_offset: bool,
):
    n_timepoints = x.shape[0]
    n_observables = Y.shape[1]

    scales = np.empty(n_observables, dtype=np.float64)
    offsets = np.empty(n_observables, dtype=np.float64)
    predicted = np.empty((n_timepoints, n_observables), dtype=np.float64)
    residuals = np.empty((n_timepoints, n_observables), dtype=np.float64)
    rss_by_observable = np.empty(n_observables, dtype=np.float64)

    all_ok = True

    sx = 0.0
    sxx = 0.0

    for i in range(n_timepoints):
        xi = x[i]
        sx += xi
        sxx += xi * xi

    for column_index in range(n_observables):
        sy = 0.0
        sxy = 0.0

        for i in range(n_timepoints):
            yi = Y[i, column_index]
            xi = x[i]

            sy += yi
            sxy += xi * yi

        if fit_scale and fit_offset:
            denominator = n_timepoints * sxx - sx * sx

            if denominator == 0.0:
                all_ok = False
                scale = 0.0
                offset = 0.0
            else:
                scale = (n_timepoints * sxy - sx * sy) / denominator
                offset = (sy - scale * sx) / n_timepoints

        elif fit_scale and not fit_offset:
            if sxx == 0.0:
                all_ok = False
                scale = 0.0
                offset = 0.0
            else:
                scale = sxy / sxx
                offset = 0.0

        elif (not fit_scale) and fit_offset:
            scale = 1.0
            offset = (sy - sx) / n_timepoints

        else:
            scale = 1.0
            offset = 0.0

        scales[column_index] = scale
        offsets[column_index] = offset

        column_rss = 0.0

        for i in range(n_timepoints):
            yi = Y[i, column_index]
            y_pred = scale * x[i] + offset
            resid = yi - y_pred

            predicted[i, column_index] = y_pred
            residuals[i, column_index] = resid
            column_rss += resid * resid

        rss_by_observable[column_index] = column_rss

    total_rss = 0.0

    for column_index in range(n_observables):
        total_rss += rss_by_observable[column_index]

    return (
        all_ok,
        scales,
        offsets,
        predicted,
        residuals,
        rss_by_observable,
        total_rss,
    )

class NumbaProjectionEngine:
    name = "numba_projection"

    def __init__(self) -> None:
        if not NUMBA_AVAILABLE:
            raise ImportError(
                "The numba_projection engine requires numba. "
                "Install numba or use engine_name='reference'."
            )

        self._fallback = ReferenceNumpyProjectionEngine()

    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            name=self.name,
            single_species_projection=True,
            multispecies_projection=True,
            supports_parallel=False,
            supports_jit=True,
            supports_gpu=False,
            supports_autodiff=False,
            notes=(
                "Numba-compiled single-species projection. "
                "Multispecies projection currently delegates to reference NumPy."
            ),
        )

    def project_single_species(
        self,
        *,
        observed_values: np.ndarray,
        species_values: np.ndarray,
        fit_scale: bool = True,
        fit_offset: bool = True,
    ) -> SingleSpeciesProjectionResult:
        y = np.asarray(observed_values, dtype=float)
        x = np.asarray(species_values, dtype=float)

        if y.shape != x.shape:
            raise ValueError(
                "observed_values and species_values must have the same shape."
            )

        ok, scale, offset = _fit_single_species_numba(
            x,
            y,
            bool(fit_scale),
            bool(fit_offset),
        )

        if not ok:
            return self._fallback.project_single_species(
                observed_values=y,
                species_values=x,
                fit_scale=fit_scale,
                fit_offset=fit_offset,
            )

        predicted = scale * x + offset
        residuals = y - predicted

        finite = np.isfinite(y) & np.isfinite(x)
        rss = float(np.nansum(residuals[finite] ** 2))

        return SingleSpeciesProjectionResult(
            scale=float(scale),
            offset=float(offset),
            predicted=predicted,
            residuals=residuals,
            rss=rss,
        )

    def project_single_species_batch(
        self,
        *,
        observed_matrix: np.ndarray,
        species_values: np.ndarray,
        fit_scale: bool = True,
        fit_offset: bool = True,
    ) -> BatchedSingleSpeciesProjectionResult:
        Y = np.asarray(observed_matrix, dtype=float)
        x = np.asarray(species_values, dtype=float)

        if Y.ndim != 2:
            raise ValueError("observed_matrix must be a 2D array.")

        if Y.shape[0] != x.shape[0]:
            raise ValueError(
                "observed_matrix row count must match species_values length."
            )

        # The fast Numba batch path assumes dense finite arrays.
        # Missing values remain supported through the reference fallback.
        if (not np.isfinite(x).all()) or (not np.isfinite(Y).all()):
            return self._fallback.project_single_species_batch(
                observed_matrix=Y,
                species_values=x,
                fit_scale=fit_scale,
                fit_offset=fit_offset,
            )

        (
            ok,
            scales,
            offsets,
            predicted,
            residuals,
            rss_by_observable,
            rss,
        ) = _fit_single_species_batch_numba(
            x,
            Y,
            bool(fit_scale),
            bool(fit_offset),
        )

        if not bool(ok):
            return self._fallback.project_single_species_batch(
                observed_matrix=Y,
                species_values=x,
                fit_scale=fit_scale,
                fit_offset=fit_offset,
            )

        return BatchedSingleSpeciesProjectionResult(
            scales=np.asarray(scales, dtype=float),
            offsets=np.asarray(offsets, dtype=float),
            predicted=np.asarray(predicted, dtype=float),
            residuals=np.asarray(residuals, dtype=float),
            rss_by_observable=np.asarray(rss_by_observable, dtype=float),
            rss=float(rss),
        )

    def project_multispecies(
        self,
        *,
        observed_values: np.ndarray,
        species_matrix: np.ndarray,
        species_names: list[str],
        fit_offset: bool = True,
    ) -> MultispeciesProjectionResult:
        return self._fallback.project_multispecies(
            observed_values=observed_values,
            species_matrix=species_matrix,
            species_names=species_names,
            fit_offset=fit_offset,
        )
