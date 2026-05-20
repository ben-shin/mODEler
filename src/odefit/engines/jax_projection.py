from __future__ import annotations

from importlib.util import find_spec

import numpy as np

from odefit.engines.base import (
    EngineCapabilities,
    MultispeciesProjectionResult,
    SingleSpeciesProjectionResult,
    BatchedSingleSpeciesProjectionResult,
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

    @jax.jit
    def _jax_batch_scale_offset(x, Y):
        n_timepoints = x.shape[0]

        sx = jnp.sum(x)
        sxx = jnp.sum(x * x)
        sy = jnp.sum(Y, axis=0)
        sxy = x @ Y

        denominator = n_timepoints * sxx - sx * sx

        scales = (n_timepoints * sxy - sx * sy) / denominator
        offsets = (sy - scales * sx) / n_timepoints

        predicted = x[:, None] * scales[None, :] + offsets[None, :]
        residuals = Y - predicted
        rss_by_observable = jnp.sum(residuals * residuals, axis=0)
        rss = jnp.sum(rss_by_observable)

        ok = jnp.isfinite(denominator) & (denominator != 0.0)

        return ok, scales, offsets, predicted, residuals, rss_by_observable, rss


    @jax.jit
    def _jax_batch_scale_only(x, Y):
        denominator = jnp.dot(x, x)

        scales = (x @ Y) / denominator
        offsets = jnp.zeros(Y.shape[1], dtype=Y.dtype)

        predicted = x[:, None] * scales[None, :]
        residuals = Y - predicted
        rss_by_observable = jnp.sum(residuals * residuals, axis=0)
        rss = jnp.sum(rss_by_observable)

        ok = jnp.isfinite(denominator) & (denominator != 0.0)

        return ok, scales, offsets, predicted, residuals, rss_by_observable, rss


    @jax.jit
    def _jax_batch_offset_only(x, Y):
        scales = jnp.ones(Y.shape[1], dtype=Y.dtype)
        offsets = jnp.mean(Y - x[:, None], axis=0)

        predicted = x[:, None] + offsets[None, :]
        residuals = Y - predicted
        rss_by_observable = jnp.sum(residuals * residuals, axis=0)
        rss = jnp.sum(rss_by_observable)

        return True, scales, offsets, predicted, residuals, rss_by_observable, rss


    @jax.jit
    def _jax_batch_fixed(x, Y):
        scales = jnp.ones(Y.shape[1], dtype=Y.dtype)
        offsets = jnp.zeros(Y.shape[1], dtype=Y.dtype)

        predicted = jnp.broadcast_to(x[:, None], Y.shape)
        residuals = Y - predicted
        rss_by_observable = jnp.sum(residuals * residuals, axis=0)
        rss = jnp.sum(rss_by_observable)

        return True, scales, offsets, predicted, residuals, rss_by_observable, rss

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
    ) -> BatchedSingleSpeciesProjectionResult:
        Y_np = np.asarray(observed_matrix, dtype=float)
        x_np = np.asarray(species_values, dtype=float)

        if Y_np.ndim != 2:
            raise ValueError("observed_matrix must be a 2D array.")

        if Y_np.shape[0] != x_np.shape[0]:
            raise ValueError(
                "observed_matrix row count must match species_values length."
            )

        # Missing values are still handled by the reference implementation.
        # The JAX fast path assumes dense finite arrays.
        if (not np.isfinite(x_np).all()) or (not np.isfinite(Y_np).all()):
            return self._fallback.project_single_species_batch(
                observed_matrix=Y_np,
                species_values=x_np,
                fit_scale=fit_scale,
                fit_offset=fit_offset,
            )

        x = jnp.asarray(x_np, dtype=jnp.float64)
        Y = jnp.asarray(Y_np, dtype=jnp.float64)

        if fit_scale and fit_offset:
            outputs = _jax_batch_scale_offset(x, Y)
        elif fit_scale and not fit_offset:
            outputs = _jax_batch_scale_only(x, Y)
        elif (not fit_scale) and fit_offset:
            outputs = _jax_batch_offset_only(x, Y)
        else:
            outputs = _jax_batch_fixed(x, Y)

        ok, scales, offsets, predicted, residuals, rss_by_observable, rss = outputs

        if not bool(np.asarray(ok)):
            return self._fallback.project_single_species_batch(
                observed_matrix=Y_np,
                species_values=x_np,
                fit_scale=fit_scale,
                fit_offset=fit_offset,
            )

        return BatchedSingleSpeciesProjectionResult(
            scales=np.asarray(scales, dtype=float),
            offsets=np.asarray(offsets, dtype=float),
            predicted=np.asarray(predicted, dtype=float),
            residuals=np.asarray(residuals, dtype=float),
            rss_by_observable=np.asarray(rss_by_observable, dtype=float),
            rss=float(np.asarray(rss)),
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
