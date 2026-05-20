from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
import json
import statistics
import time
from typing import Any

import numpy as np
import pandas as pd


JAX_AVAILABLE = find_spec("jax") is not None and find_spec("jax.numpy") is not None

if JAX_AVAILABLE:
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
else:
    jax = None
    jnp = None


@dataclass
class BatchedProjectionBenchmarkResult:
    name: str
    n_timepoints: int
    n_observables: int
    n_repeats: int
    times_seconds: list[float]
    metadata: dict[str, Any]

    @property
    def min_seconds(self) -> float:
        return min(self.times_seconds)

    @property
    def max_seconds(self) -> float:
        return max(self.times_seconds)

    @property
    def mean_seconds(self) -> float:
        return statistics.mean(self.times_seconds)

    @property
    def median_seconds(self) -> float:
        return statistics.median(self.times_seconds)

    @property
    def stdev_seconds(self) -> float:
        if len(self.times_seconds) < 2:
            return 0.0

        return statistics.stdev(self.times_seconds)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n_timepoints": self.n_timepoints,
            "n_observables": self.n_observables,
            "n_repeats": self.n_repeats,
            "times_seconds": self.times_seconds,
            "min_seconds": self.min_seconds,
            "max_seconds": self.max_seconds,
            "mean_seconds": self.mean_seconds,
            "median_seconds": self.median_seconds,
            "stdev_seconds": self.stdev_seconds,
            "metadata": self.metadata,
        }


def make_batched_single_species_projection_data(
    *,
    n_timepoints: int = 50,
    n_observables: int = 1000,
    k: float = 0.4,
    noise: float = 0.002,
    random_seed: int = 123,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build synthetic batched projection data.

    Returns:
        species_values:
            shape (n_timepoints,)
        observed_matrix:
            shape (n_timepoints, n_observables)

    Each observable is:

        y_i(t) = scale_i * species(t) + offset_i + noise
    """

    rng = np.random.default_rng(random_seed)

    timepoints = np.linspace(0.0, 10.0, n_timepoints)
    species_values = np.exp(-k * timepoints)

    scales = rng.uniform(0.5, 2.0, size=n_observables)
    offsets = rng.uniform(-0.1, 0.1, size=n_observables)

    observed_matrix = (
        species_values[:, None] * scales[None, :]
        + offsets[None, :]
        + rng.normal(0.0, noise, size=(n_timepoints, n_observables))
    )

    return species_values, observed_matrix


def numpy_loop_batched_single_species_projection(
    *,
    species_values: np.ndarray,
    observed_matrix: np.ndarray,
    fit_scale: bool = True,
    fit_offset: bool = True,
) -> dict[str, np.ndarray | float]:
    """
    Baseline Python/NumPy loop over observable columns.
    """

    x = np.asarray(species_values, dtype=float)
    Y = np.asarray(observed_matrix, dtype=float)

    n_observables = Y.shape[1]

    scales = np.empty(n_observables, dtype=float)
    offsets = np.empty(n_observables, dtype=float)
    predicted = np.empty_like(Y, dtype=float)
    residuals = np.empty_like(Y, dtype=float)
    rss = np.empty(n_observables, dtype=float)

    for column_index in range(n_observables):
        y = Y[:, column_index]
        finite = np.isfinite(x) & np.isfinite(y)

        x_fit = x[finite]
        y_fit = y[finite]

        if fit_scale and fit_offset:
            design = np.column_stack([x_fit, np.ones(len(x_fit))])
            beta, *_ = np.linalg.lstsq(design, y_fit, rcond=None)
            scale = float(beta[0])
            offset = float(beta[1])
        elif fit_scale and not fit_offset:
            denominator = float(np.dot(x_fit, x_fit))
            if denominator == 0.0:
                raise ValueError("Cannot fit scale: species vector has zero norm.")
            scale = float(np.dot(x_fit, y_fit) / denominator)
            offset = 0.0
        elif not fit_scale and fit_offset:
            scale = 1.0
            offset = float(np.mean(y_fit - x_fit))
        else:
            scale = 1.0
            offset = 0.0

        y_pred = scale * x + offset
        y_resid = y - y_pred

        scales[column_index] = scale
        offsets[column_index] = offset
        predicted[:, column_index] = y_pred
        residuals[:, column_index] = y_resid
        rss[column_index] = float(np.nansum(y_resid[finite] ** 2))

    return {
        "scales": scales,
        "offsets": offsets,
        "predicted": predicted,
        "residuals": residuals,
        "rss_by_observable": rss,
        "rss": float(np.sum(rss)),
    }


def numpy_vectorized_batched_single_species_projection(
    *,
    species_values: np.ndarray,
    observed_matrix: np.ndarray,
    fit_scale: bool = True,
    fit_offset: bool = True,
) -> dict[str, np.ndarray | float]:
    """
    Vectorized NumPy batched projection.

    This is the main CPU baseline that JAX/GPU must beat.
    It assumes no missing values for the fast path.
    """

    x = np.asarray(species_values, dtype=float)
    Y = np.asarray(observed_matrix, dtype=float)

    if np.isnan(x).any() or np.isnan(Y).any():
        return numpy_loop_batched_single_species_projection(
            species_values=x,
            observed_matrix=Y,
            fit_scale=fit_scale,
            fit_offset=fit_offset,
        )

    n_timepoints = x.shape[0]
    n_observables = Y.shape[1]

    if fit_scale and fit_offset:
        sx = float(np.sum(x))
        sxx = float(np.sum(x * x))
        sy = np.sum(Y, axis=0)
        sxy = x @ Y

        denominator = n_timepoints * sxx - sx * sx

        if denominator == 0.0:
            raise ValueError("Cannot fit scale/offset: singular design.")

        scales = (n_timepoints * sxy - sx * sy) / denominator
        offsets = (sy - scales * sx) / n_timepoints

    elif fit_scale and not fit_offset:
        denominator = float(np.dot(x, x))

        if denominator == 0.0:
            raise ValueError("Cannot fit scale: species vector has zero norm.")

        scales = (x @ Y) / denominator
        offsets = np.zeros(n_observables, dtype=float)

    elif not fit_scale and fit_offset:
        scales = np.ones(n_observables, dtype=float)
        offsets = np.mean(Y - x[:, None], axis=0)

    else:
        scales = np.ones(n_observables, dtype=float)
        offsets = np.zeros(n_observables, dtype=float)

    predicted = x[:, None] * scales[None, :] + offsets[None, :]
    residuals = Y - predicted
    rss_by_observable = np.sum(residuals**2, axis=0)

    return {
        "scales": scales,
        "offsets": offsets,
        "predicted": predicted,
        "residuals": residuals,
        "rss_by_observable": rss_by_observable,
        "rss": float(np.sum(rss_by_observable)),
    }


if JAX_AVAILABLE:

    @jax.jit
    def _jax_vectorized_scale_offset(x, Y):
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

        return scales, offsets, predicted, residuals, rss_by_observable, rss


    @jax.jit
    def _jax_vectorized_scale_only(x, Y):
        denominator = jnp.dot(x, x)

        scales = (x @ Y) / denominator
        offsets = jnp.zeros(Y.shape[1], dtype=Y.dtype)

        predicted = x[:, None] * scales[None, :]
        residuals = Y - predicted
        rss_by_observable = jnp.sum(residuals * residuals, axis=0)
        rss = jnp.sum(rss_by_observable)

        return scales, offsets, predicted, residuals, rss_by_observable, rss


    @jax.jit
    def _jax_vectorized_offset_only(x, Y):
        scales = jnp.ones(Y.shape[1], dtype=Y.dtype)
        offsets = jnp.mean(Y - x[:, None], axis=0)

        predicted = x[:, None] + offsets[None, :]
        residuals = Y - predicted
        rss_by_observable = jnp.sum(residuals * residuals, axis=0)
        rss = jnp.sum(rss_by_observable)

        return scales, offsets, predicted, residuals, rss_by_observable, rss


    @jax.jit
    def _jax_vectorized_fixed(x, Y):
        scales = jnp.ones(Y.shape[1], dtype=Y.dtype)
        offsets = jnp.zeros(Y.shape[1], dtype=Y.dtype)

        predicted = x[:, None]
        residuals = Y - predicted
        rss_by_observable = jnp.sum(residuals * residuals, axis=0)
        rss = jnp.sum(rss_by_observable)

        return scales, offsets, predicted, residuals, rss_by_observable, rss


def jax_vectorized_batched_single_species_projection(
    *,
    species_values: np.ndarray,
    observed_matrix: np.ndarray,
    fit_scale: bool = True,
    fit_offset: bool = True,
) -> dict[str, np.ndarray | float]:
    """
    Vectorized JAX batched projection.

    This is intentionally benchmark-only for now.
    It assumes no missing values.
    """

    if not JAX_AVAILABLE:
        raise ImportError("JAX is not installed.")

    x = jnp.asarray(np.asarray(species_values, dtype=float), dtype=jnp.float64)
    Y = jnp.asarray(np.asarray(observed_matrix, dtype=float), dtype=jnp.float64)

    if fit_scale and fit_offset:
        outputs = _jax_vectorized_scale_offset(x, Y)
    elif fit_scale and not fit_offset:
        outputs = _jax_vectorized_scale_only(x, Y)
    elif not fit_scale and fit_offset:
        outputs = _jax_vectorized_offset_only(x, Y)
    else:
        outputs = _jax_vectorized_fixed(x, Y)

    scales, offsets, predicted, residuals, rss_by_observable, rss = outputs

    return {
        "scales": np.asarray(scales),
        "offsets": np.asarray(offsets),
        "predicted": np.asarray(predicted),
        "residuals": np.asarray(residuals),
        "rss_by_observable": np.asarray(rss_by_observable),
        "rss": float(np.asarray(rss)),
    }


def _time_function(function, *, n_repeats: int) -> list[float]:
    times = []

    for _ in range(n_repeats):
        start = time.perf_counter()
        function()
        end = time.perf_counter()
        times.append(end - start)

    return times


def benchmark_batched_projection_method(
    *,
    name: str,
    function,
    species_values: np.ndarray,
    observed_matrix: np.ndarray,
    n_repeats: int,
    warmup: bool = True,
    metadata: dict[str, Any] | None = None,
) -> BatchedProjectionBenchmarkResult:
    if warmup:
        function(
            species_values=species_values,
            observed_matrix=observed_matrix,
        )

    times = _time_function(
        lambda: function(
            species_values=species_values,
            observed_matrix=observed_matrix,
        ),
        n_repeats=n_repeats,
    )

    return BatchedProjectionBenchmarkResult(
        name=name,
        n_timepoints=species_values.shape[0],
        n_observables=observed_matrix.shape[1],
        n_repeats=n_repeats,
        times_seconds=times,
        metadata=metadata or {},
    )


def run_batched_projection_benchmarks(
    *,
    n_timepoints: int = 50,
    n_observables: int = 1000,
    n_repeats: int = 100,
    include_loop: bool = True,
    include_jax: bool = True,
) -> list[BatchedProjectionBenchmarkResult]:
    species_values, observed_matrix = make_batched_single_species_projection_data(
        n_timepoints=n_timepoints,
        n_observables=n_observables,
    )

    results = []

    if include_loop:
        results.append(
            benchmark_batched_projection_method(
                name="numpy_loop",
                function=numpy_loop_batched_single_species_projection,
                species_values=species_values,
                observed_matrix=observed_matrix,
                n_repeats=n_repeats,
                metadata={"backend": "numpy", "vectorized": False},
            )
        )

    results.append(
        benchmark_batched_projection_method(
            name="numpy_vectorized",
            function=numpy_vectorized_batched_single_species_projection,
            species_values=species_values,
            observed_matrix=observed_matrix,
            n_repeats=n_repeats,
            metadata={"backend": "numpy", "vectorized": True},
        )
    )

    if include_jax and JAX_AVAILABLE:
        results.append(
            benchmark_batched_projection_method(
                name="jax_vectorized",
                function=jax_vectorized_batched_single_species_projection,
                species_values=species_values,
                observed_matrix=observed_matrix,
                n_repeats=n_repeats,
                metadata={
                    "backend": "jax",
                    "vectorized": True,
                    "jit": True,
                },
            )
        )

    return results


def assert_projection_outputs_close(
    left: dict[str, np.ndarray | float],
    right: dict[str, np.ndarray | float],
    *,
    rtol: float = 1e-7,
    atol: float = 1e-9,
) -> None:
    for key in [
        "scales",
        "offsets",
        "predicted",
        "residuals",
        "rss_by_observable",
    ]:
        np.testing.assert_allclose(
            left[key],
            right[key],
            rtol=rtol,
            atol=atol,
        )

    assert np.isclose(
        float(left["rss"]),
        float(right["rss"]),
        rtol=rtol,
        atol=atol,
    )


def export_batched_projection_benchmarks(
    results: list[BatchedProjectionBenchmarkResult],
    output_dir: str | Path,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    json_path = output_path / "batched_projection_benchmarks.json"
    csv_path = output_path / "batched_projection_benchmarks.csv"

    payload = [result.to_dict() for result in results]

    with json_path.open("w") as handle:
        json.dump(payload, handle, indent=2)

    rows = []

    for result in results:
        row = {
            "name": result.name,
            "n_timepoints": result.n_timepoints,
            "n_observables": result.n_observables,
            "n_repeats": result.n_repeats,
            "min_seconds": result.min_seconds,
            "max_seconds": result.max_seconds,
            "mean_seconds": result.mean_seconds,
            "median_seconds": result.median_seconds,
            "stdev_seconds": result.stdev_seconds,
        }

        for key, value in result.metadata.items():
            row[f"metadata_{key}"] = value

        rows.append(row)

    pd.DataFrame(rows).to_csv(csv_path, index=False)

    return {
        "json": json_path,
        "csv": csv_path,
    }
