from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EngineCapabilities:
    name: str
    ode_solver: bool = False
    least_squares: bool = False
    single_species_projection: bool = False
    multispecies_projection: bool = False
    supports_parallel: bool = False
    supports_jit: bool = False
    supports_gpu: bool = False
    supports_autodiff: bool = False
    notes: str = ""


@dataclass
class SolverEngineResult:
    success: bool
    message: str
    timepoints: np.ndarray
    species: list[str]
    values: dict[str, np.ndarray]

    def to_dataframe(self) -> pd.DataFrame:
        dataframe = pd.DataFrame({"time": self.timepoints})

        for species_name in self.species:
            dataframe[species_name] = self.values[species_name]

        return dataframe


@dataclass
class SingleSpeciesProjectionResult:
    scale: float
    offset: float
    predicted: np.ndarray
    residuals: np.ndarray
    rss: float


@dataclass
class MultispeciesProjectionResult:
    coefficients: dict[str, float]
    offset: float
    predicted: np.ndarray
    residuals: np.ndarray
    rss: float


@dataclass
class LeastSquaresEngineResult:
    success: bool
    message: str
    x: np.ndarray
    cost: float
    fun: np.ndarray
    raw_result: Any


class SolverEngine(Protocol):
    name: str

    def capabilities(self) -> EngineCapabilities:
        ...

    def solve(
        self,
        *,
        model: Any,
        parameters: dict[str, float],
        initial_conditions: dict[str, float],
        timepoints: np.ndarray,
        settings: Any | None = None,
    ) -> SolverEngineResult:
        ...


class ProjectionEngine(Protocol):
    name: str

    def capabilities(self) -> EngineCapabilities:
        ...

    def project_single_species(
        self,
        *,
        observed_values: np.ndarray,
        species_values: np.ndarray,
        fit_scale: bool = True,
        fit_offset: bool = True,
    ) -> SingleSpeciesProjectionResult:
        ...

    def project_multispecies(
        self,
        *,
        observed_values: np.ndarray,
        species_matrix: np.ndarray,
        species_names: list[str],
        fit_offset: bool = True,
    ) -> MultispeciesProjectionResult:
        ...


class LeastSquaresEngine(Protocol):
    name: str

    def capabilities(self) -> EngineCapabilities:
        ...

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
        ...


@dataclass
class BackendEngineBundle:
    name: str
    solver: SolverEngine
    projection: ProjectionEngine
    least_squares: LeastSquaresEngine

    def capabilities(self) -> dict[str, EngineCapabilities]:
        return {
            "solver": self.solver.capabilities(),
            "projection": self.projection.capabilities(),
            "least_squares": self.least_squares.capabilities(),
        }
