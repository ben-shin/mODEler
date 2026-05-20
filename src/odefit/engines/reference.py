from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import least_squares as scipy_least_squares

from odefit.engines.base import (
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
            raw_result=result,
        )
