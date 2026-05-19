from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from odefit.model.model_spec import ModelSpec
from odefit.performance.array_rhs import (
    CompiledMassActionModel,
    compile_mass_action_model,
    concentration_dict_to_array,
    evaluate_mass_action_rhs,
    parameter_dict_to_array,
)
from odefit.performance.numba_rhs import (
    evaluate_mass_action_rhs_numba,
    is_numba_available,
    warm_up_numba_rhs,
)


@dataclass
class ArraySolveResult:
    """
    Result from array-based solve_ivp simulation.
    """

    timepoints: np.ndarray
    species: list[str]
    values: np.ndarray
    success: bool
    message: str
    backend: str
    method: str
    nfev: int

    def get_species_values(
        self,
        species_name: str,
    ) -> np.ndarray:
        """
        Return simulated values for one species.
        """

        if species_name not in self.species:
            raise ValueError(f"Species not found in result: {species_name}")

        species_index = self.species.index(species_name)

        return self.values[:, species_index]


def _as_compiled_model(
    model: ModelSpec | CompiledMassActionModel,
) -> CompiledMassActionModel:
    """
    Convert ModelSpec to CompiledMassActionModel if needed.
    """

    if isinstance(model, CompiledMassActionModel):
        return model

    return compile_mass_action_model(model)


def _parameters_to_array(
    compiled_model: CompiledMassActionModel,
    parameters: dict[str, float] | np.ndarray,
) -> np.ndarray:
    """
    Convert parameters to compiled-model array order.
    """

    if isinstance(parameters, dict):
        return parameter_dict_to_array(
            compiled_model=compiled_model,
            parameter_values=parameters,
        )

    parameter_array = np.asarray(parameters, dtype=float)

    if parameter_array.shape != (compiled_model.n_parameters,):
        raise ValueError(
            f"parameters must have shape ({compiled_model.n_parameters},), "
            f"got {parameter_array.shape}"
        )

    return parameter_array


def _initial_conditions_to_array(
    compiled_model: CompiledMassActionModel,
    initial_conditions: dict[str, float] | np.ndarray,
) -> np.ndarray:
    """
    Convert initial conditions to compiled-model species order.
    """

    if isinstance(initial_conditions, dict):
        return concentration_dict_to_array(
            compiled_model=compiled_model,
            concentrations=initial_conditions,
        )

    initial_array = np.asarray(initial_conditions, dtype=float)

    if initial_array.shape != (compiled_model.n_species,):
        raise ValueError(
            f"initial_conditions must have shape ({compiled_model.n_species},), "
            f"got {initial_array.shape}"
        )

    return initial_array


def _validate_timepoints(
    timepoints: np.ndarray,
) -> np.ndarray:
    """
    Validate and return timepoint array.
    """

    time_array = np.asarray(timepoints, dtype=float)

    if time_array.ndim != 1:
        raise ValueError("timepoints must be a one-dimensional array")

    if len(time_array) < 2:
        raise ValueError("timepoints must contain at least two values")

    if np.any(np.diff(time_array) <= 0.0):
        raise ValueError("timepoints must be strictly increasing")

    return time_array


def make_array_rhs_function(
    compiled_model: CompiledMassActionModel,
    parameter_array: np.ndarray,
    backend: str = "numpy",
    clip_negative_concentrations: bool = False,
):
    """
    Build a solve_ivp-compatible RHS function.

    Supported backends:
    - numpy
    - numba
    """

    backend = backend.lower()

    if backend == "numpy":

        def rhs(_time, concentrations):
            return evaluate_mass_action_rhs(
                compiled_model=compiled_model,
                concentrations=concentrations,
                parameters=parameter_array,
                clip_negative_concentrations=clip_negative_concentrations,
            )

        return rhs

    if backend == "numba":
        if not is_numba_available():
            raise RuntimeError(
                "Numba backend requested but numba is not installed."
            )

        def rhs(_time, concentrations):
            return evaluate_mass_action_rhs_numba(
                compiled_model=compiled_model,
                concentrations=concentrations,
                parameters=parameter_array,
                clip_negative_concentrations=clip_negative_concentrations,
            )

        return rhs

    raise ValueError(f"Unknown array RHS backend: {backend}")


def solve_array_mass_action_model(
    model: ModelSpec | CompiledMassActionModel,
    parameters: dict[str, float] | np.ndarray,
    initial_conditions: dict[str, float] | np.ndarray,
    timepoints: np.ndarray,
    backend: str = "numpy",
    method: str = "LSODA",
    rtol: float = 1e-6,
    atol: float = 1e-9,
    clip_negative_concentrations: bool = False,
) -> ArraySolveResult:
    """
    Simulate a mass-action model using array-based RHS and scipy.solve_ivp.

    This is a performance prototype backend. It does not replace the main
    simulation API yet.
    """

    compiled_model = _as_compiled_model(model)

    time_array = _validate_timepoints(timepoints)

    parameter_array = _parameters_to_array(
        compiled_model=compiled_model,
        parameters=parameters,
    )

    initial_array = _initial_conditions_to_array(
        compiled_model=compiled_model,
        initial_conditions=initial_conditions,
    )

    backend = backend.lower()

    if backend == "numba":
        if not is_numba_available():
            raise RuntimeError(
                "Numba backend requested but numba is not installed."
            )

        warm_up_numba_rhs(
            compiled_model=compiled_model,
            concentrations=initial_array,
            parameters=parameter_array,
        )

    rhs = make_array_rhs_function(
        compiled_model=compiled_model,
        parameter_array=parameter_array,
        backend=backend,
        clip_negative_concentrations=clip_negative_concentrations,
    )

    solution = solve_ivp(
        fun=rhs,
        t_span=(float(time_array[0]), float(time_array[-1])),
        y0=initial_array,
        t_eval=time_array,
        method=method,
        rtol=rtol,
        atol=atol,
    )

    values = solution.y.T

    return ArraySolveResult(
        timepoints=time_array,
        species=compiled_model.species,
        values=values,
        success=bool(solution.success),
        message=str(solution.message),
        backend=backend,
        method=method,
        nfev=int(solution.nfev),
    )
