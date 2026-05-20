from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import least_squares as scipy_least_squares

from odefit.engines.base import (
    BatchedSingleSpeciesProjectionResult,
    EngineCapabilities,
    LeastSquaresEngineResult,
    MultispeciesProjectionResult,
    SingleSpeciesProjectionResult,
    SolverEngineResult,
)
from odefit.simulation.solver import simulate_model


class ReferenceScipySolverEngine:
    name = "reference_scipy_solver"

    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            name=self.name,
            ode_solver=True,
            supports_parallel=False,
            supports_jit=False,
            supports_gpu=False,
            supports_autodiff=False,
            notes="Reference SciPy-based ODE solver engine.",
        )

    def solve(
        self,
        *,
        model: Any,
        parameters: dict[str, float],
        initial_conditions: dict[str, float],
        timepoints: np.ndarray,
        settings: Any | None = None,
    ) -> SolverEngineResult:
        kwargs = {
            "model": model,
            "parameters": parameters,
            "initial_conditions": initial_conditions,
            "timepoints": timepoints,
        }

        if settings is not None:
            kwargs["settings"] = settings

        result = simulate_model(**kwargs)

        values = {
            species_name: np.asarray(
                result.get_species_values(species_name),
                dtype=float,
            )
            for species_name in result.species
        }

        return SolverEngineResult(
            success=bool(result.success),
            message=str(result.message),
            timepoints=np.asarray(result.timepoints, dtype=float),
            species=list(result.species),
            values=values,
        )


class ReferenceNumpyProjectionEngine:
    name = "reference_numpy_projection"

    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            name=self.name,
            single_species_projection=True,
            multispecies_projection=True,
            supports_parallel=False,
            supports_jit=False,
            supports_gpu=False,
            supports_autodiff=False,
            notes="Reference NumPy least-squares projection engine.",
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

        finite = np.isfinite(y) & np.isfinite(x)

        if not finite.any():
            raise ValueError("No finite values available for projection.")

        x_fit = x[finite]
        y_fit = y[finite]

        if fit_scale and fit_offset:
            design = np.column_stack([x_fit, np.ones(len(x_fit))])
            beta, *_ = np.linalg.lstsq(design, y_fit, rcond=None)
            scale = float(beta[0])
            offset = float(beta[1])
        elif fit_scale and not fit_offset:
            denom = float(np.dot(x_fit, x_fit))

            if denom == 0.0:
                raise ValueError("Cannot fit scale: species vector has zero norm.")

            scale = float(np.dot(x_fit, y_fit) / denom)
            offset = 0.0
        elif not fit_scale and fit_offset:
            scale = 1.0
            offset = float(np.mean(y_fit - x_fit))
        else:
            scale = 1.0
            offset = 0.0

        predicted = scale * x + offset
        residuals = y - predicted
        rss = float(np.nansum(residuals[finite] ** 2))

        return SingleSpeciesProjectionResult(
            scale=scale,
            offset=offset,
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
        y_matrix = np.asarray(observed_matrix, dtype=float)
        x = np.asarray(species_values, dtype=float)

        if y_matrix.ndim != 2:
            raise ValueError("observed_matrix must be a 2D array.")

        if y_matrix.shape[0] != x.shape[0]:
            raise ValueError(
                "observed_matrix row count must match species_values length."
            )

        n_observables = y_matrix.shape[1]

        scales = np.empty(n_observables, dtype=float)
        offsets = np.empty(n_observables, dtype=float)
        predicted = np.empty_like(y_matrix, dtype=float)
        residuals = np.empty_like(y_matrix, dtype=float)
        rss_by_observable = np.empty(n_observables, dtype=float)

        # Fast path: no missing values.
        if np.isfinite(x).all() and np.isfinite(y_matrix).all():
            n_timepoints = x.shape[0]

            if fit_scale and fit_offset:
                sx = float(np.sum(x))
                sxx = float(np.sum(x * x))
                sy = np.sum(y_matrix, axis=0)
                sxy = x @ y_matrix

                denominator = n_timepoints * sxx - sx * sx

                if denominator == 0.0:
                    raise ValueError("Cannot fit scale/offset: singular design.")

                scales = (n_timepoints * sxy - sx * sy) / denominator
                offsets = (sy - scales * sx) / n_timepoints

            elif fit_scale and not fit_offset:
                denominator = float(np.dot(x, x))

                if denominator == 0.0:
                    raise ValueError(
                        "Cannot fit scale: species vector has zero norm."
                    )

                scales = (x @ y_matrix) / denominator
                offsets = np.zeros(n_observables, dtype=float)

            elif not fit_scale and fit_offset:
                scales = np.ones(n_observables, dtype=float)
                offsets = np.mean(y_matrix - x[:, None], axis=0)

            else:
                scales = np.ones(n_observables, dtype=float)
                offsets = np.zeros(n_observables, dtype=float)

            predicted = x[:, None] * scales[None, :] + offsets[None, :]
            residuals = y_matrix - predicted
            rss_by_observable = np.sum(residuals**2, axis=0)

            return BatchedSingleSpeciesProjectionResult(
                scales=np.asarray(scales, dtype=float),
                offsets=np.asarray(offsets, dtype=float),
                predicted=predicted,
                residuals=residuals,
                rss_by_observable=rss_by_observable,
                rss=float(np.sum(rss_by_observable)),
            )

        # Robust missing-value fallback.
        for column_index in range(n_observables):
            result = self.project_single_species(
                observed_values=y_matrix[:, column_index],
                species_values=x,
                fit_scale=fit_scale,
                fit_offset=fit_offset,
            )

            scales[column_index] = result.scale
            offsets[column_index] = result.offset
            predicted[:, column_index] = result.predicted
            residuals[:, column_index] = result.residuals
            rss_by_observable[column_index] = result.rss

        return BatchedSingleSpeciesProjectionResult(
            scales=scales,
            offsets=offsets,
            predicted=predicted,
            residuals=residuals,
            rss_by_observable=rss_by_observable,
            rss=float(np.sum(rss_by_observable)),
        )

    def project_multispecies(
        self,
        *,
        observed_values: np.ndarray,
        species_matrix: np.ndarray,
        species_names: list[str],
        fit_offset: bool = True,
    ) -> MultispeciesProjectionResult:
        y = np.asarray(observed_values, dtype=float)
        X = np.asarray(species_matrix, dtype=float)

        if X.ndim != 2:
            raise ValueError("species_matrix must be a 2D array.")

        if X.shape[1] != len(species_names):
            raise ValueError(
                "species_matrix column count must match species_names length."
            )

        finite = np.isfinite(y) & np.all(np.isfinite(X), axis=1)

        if not finite.any():
            raise ValueError("No finite values available for multispecies projection.")

        X_fit = X[finite]
        y_fit = y[finite]

        if fit_offset:
            design = np.column_stack([X_fit, np.ones(len(X_fit))])
        else:
            design = X_fit

        beta, *_ = np.linalg.lstsq(design, y_fit, rcond=None)

        if fit_offset:
            coefficient_values = beta[:-1]
            offset = float(beta[-1])
        else:
            coefficient_values = beta
            offset = 0.0

        coefficients = {
            species_name: float(value)
            for species_name, value in zip(species_names, coefficient_values)
        }

        predicted = X @ coefficient_values + offset
        residuals = y - predicted
        rss = float(np.nansum(residuals[finite] ** 2))

        return MultispeciesProjectionResult(
            coefficients=coefficients,
            offset=offset,
            predicted=predicted,
            residuals=residuals,
            rss=rss,
        )


class ReferenceScipyLeastSquaresEngine:
    name = "reference_scipy_least_squares"

    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            name=self.name,
            least_squares=True,
            supports_parallel=False,
            supports_jit=False,
            supports_gpu=False,
            supports_autodiff=False,
            notes="Reference SciPy least_squares optimizer engine.",
        )

    def least_squares(
        self,
        residual_function,
        *,
        x0: np.ndarray,
        bounds: tuple[np.ndarray, np.ndarray] | None = None,
        method: str = "trf",
        loss: str = "linear",
        max_nfev: int | None = None,
    ) -> LeastSquaresEngineResult:
        if bounds is None:
            bounds = (
                np.full_like(x0, -np.inf, dtype=float),
                np.full_like(x0, np.inf, dtype=float),
            )

        result = scipy_least_squares(
            residual_function,
            x0=np.asarray(x0, dtype=float),
            bounds=bounds,
            method=method,
            loss=loss,
            max_nfev=max_nfev,
        )

        return LeastSquaresEngineResult(
            success=bool(result.success),
            message=str(result.message),
            x=np.asarray(result.x, dtype=float),
            cost=float(result.cost),
            fun=np.asarray(result.fun, dtype=float),
            nfev=int(result.nfev),
            status=int(result.status),
            optimality=float(result.optimality),
            active_mask=np.asarray(result.active_mask),
            raw_result=result,
        )
