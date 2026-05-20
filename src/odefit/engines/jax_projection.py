from __future__ import annotations

from importlib.util import find_spec

import numpy as np

from odefit.engines.base import (
    EngineCapabilities,
    MultispeciesProjectionResult,
    SingleSpeciesProjectionResult,
)
from odefit.engines.reference import ReferenceNumpyProjectionEngine


JAX_AVAILABLE = find_spec("jax") is not None and find_spec("jax.numpy") is not None

if JAX_AVAILABLE:
    import jax
    import jax.numpy as jnp

    # Use float64 so results match NumPy/SciPy reference more closely.
    jax.config.update("jax_enable_x64", True)
else:
    jax = None
    jnp = None


def is_jax_available() -> bool:
    return JAX_AVAILABLE


if JAX_AVAILABLE:

    @jax.jit
    def _fit_single_species_scale_offset_jax(x, y):
        finite = jnp.isfinite(x) & jnp.isfinite(y)
        weights = finite.astype(jnp.float64)

        n = jnp.sum(weights)
        sx = jnp.sum(weights * x)
        sy = jnp.sum(weights * y)
        sxx = jnp.sum(weights * x * x)
        sxy = jnp.sum(weights * x * y)

        denominator = n * sxx - sx * sx

        scale = (n * sxy - sx * sy) / denominator
        offset = (sy - scale * sx) / n

        ok = (n >= 2.0) & jnp.isfinite(denominator) & (denominator != 0.0)

        return ok, scale, offset


    @jax.jit
    def _fit_single_species_scale_only_jax(x, y):
        finite = jnp.isfinite(x) & jnp.isfinite(y)
        weights = finite.astype(jnp.float64)

        sxx = jnp.sum(weights * x * x)
        sxy = jnp.sum(weights * x * y)

        scale = sxy / sxx
        offset = 0.0

        ok = jnp.isfinite(sxx) & (sxx != 0.0)

        return ok, scale, offset


    @jax.jit
    def _fit_single_species_offset_only_jax(x, y):
        finite = jnp.isfinite(x) & jnp.isfinite(y)
        weights = finite.astype(jnp.float64)

        n = jnp.sum(weights)
        sx = jnp.sum(weights * x)
        sy = jnp.sum(weights * y)

        scale = 1.0
        offset = (sy - sx) / n

        ok = n >= 1.0

        return ok, scale, offset


    @jax.jit
    def _fit_single_species_fixed_jax(x, y):
        finite = jnp.isfinite(x) & jnp.isfinite(y)
        n = jnp.sum(finite.astype(jnp.float64))

        return n >= 1.0, 1.0, 0.0


class JaxProjectionEngine:
    name = "jax_projection"

    def __init__(self) -> None:
        if not JAX_AVAILABLE:
            raise ImportError(
                "The jax_projection engine requires jax. "
                "Install jax or use engine_name='reference'."
            )

        self._fallback = ReferenceNumpyProjectionEngine()

    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            name=self.name,
            single_species_projection=True,
            multispecies_projection=True,
            supports_parallel=False,
            supports_jit=True,
            supports_gpu=True,
            supports_autodiff=True,
            notes=(
                "JAX single-species projection kernel. "
                "Multispecies projection currently delegates to reference NumPy. "
                "ODE solving and nonlinear least-squares remain reference SciPy."
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
        y_np = np.asarray(observed_values, dtype=float)
        x_np = np.asarray(species_values, dtype=float)

        if y_np.shape != x_np.shape:
            raise ValueError(
                "observed_values and species_values must have the same shape."
            )

        x = jnp.asarray(x_np, dtype=jnp.float64)
        y = jnp.asarray(y_np, dtype=jnp.float64)

        if fit_scale and fit_offset:
            ok, scale, offset = _fit_single_species_scale_offset_jax(x, y)
        elif fit_scale and not fit_offset:
            ok, scale, offset = _fit_single_species_scale_only_jax(x, y)
        elif (not fit_scale) and fit_offset:
            ok, scale, offset = _fit_single_species_offset_only_jax(x, y)
        else:
            ok, scale, offset = _fit_single_species_fixed_jax(x, y)

        ok_bool = bool(np.asarray(ok))

        if not ok_bool:
            return self._fallback.project_single_species(
                observed_values=y_np,
                species_values=x_np,
                fit_scale=fit_scale,
                fit_offset=fit_offset,
            )

        scale_float = float(np.asarray(scale))
        offset_float = float(np.asarray(offset))

        predicted = scale_float * x_np + offset_float
        residuals = y_np - predicted

        finite = np.isfinite(y_np) & np.isfinite(x_np)
        rss = float(np.nansum(residuals[finite] ** 2))

        return SingleSpeciesProjectionResult(
            scale=scale_float,
            offset=offset_float,
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
