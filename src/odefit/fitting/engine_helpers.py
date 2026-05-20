from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from odefit.engines.base import BackendEngineBundle
from odefit.engines.registry import get_engine_bundle


def resolve_engine_bundle(
    *,
    engine_name: str = "reference",
    engine_bundle: BackendEngineBundle | None = None,
) -> BackendEngineBundle:
    if engine_bundle is not None:
        return engine_bundle

    return get_engine_bundle(engine_name)


def engine_solve_to_dataframe(
    *,
    engine_bundle: BackendEngineBundle,
    model: Any,
    parameters: dict[str, float],
    initial_conditions: dict[str, float],
    timepoints,
    settings: Any | None = None,
) -> pd.DataFrame:
    result = engine_bundle.solver.solve(
        model=model,
        parameters=parameters,
        initial_conditions=initial_conditions,
        timepoints=np.asarray(timepoints, dtype=float),
        settings=settings,
    )

    if not result.success:
        raise RuntimeError(result.message)

    return result.to_dataframe()


def engine_project_single_species(
    *,
    engine_bundle: BackendEngineBundle,
    observed_values,
    species_values,
    fit_scale: bool = True,
    fit_offset: bool = True,
):
    return engine_bundle.projection.project_single_species(
        observed_values=np.asarray(observed_values, dtype=float),
        species_values=np.asarray(species_values, dtype=float),
        fit_scale=fit_scale,
        fit_offset=fit_offset,
    )


def engine_project_multispecies(
    *,
    engine_bundle: BackendEngineBundle,
    observed_values,
    species_matrix,
    species_names: list[str],
    fit_offset: bool = True,
):
    return engine_bundle.projection.project_multispecies(
        observed_values=np.asarray(observed_values, dtype=float),
        species_matrix=np.asarray(species_matrix, dtype=float),
        species_names=species_names,
        fit_offset=fit_offset,
    )


def engine_least_squares(
    *,
    engine_bundle: BackendEngineBundle,
    residual_function,
    x0,
    bounds,
    method: str = "trf",
    loss: str = "linear",
    max_nfev: int | None = None,
):
    return engine_bundle.least_squares.least_squares(
        residual_function,
        x0=np.asarray(x0, dtype=float),
        bounds=bounds,
        method=method,
        loss=loss,
        max_nfev=max_nfev,
    )
