from __future__ import annotations

from importlib.util import find_spec

import numpy as np

from odefit.engines.base import (
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
    ):
        return self._fallback.project_single_species_batch(
            observed_matrix=observed_matrix,
            species_values=species_values,
            fit_scale=fit_scale,
            fit_offset=fit_offset,
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
