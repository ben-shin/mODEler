from __future__ import annotations

import importlib.util

import numpy as np

from odefit.performance.array_rhs import CompiledMassActionModel


def is_numba_available() -> bool:
    """
    Return True if numba is installed.
    """

    return importlib.util.find_spec("numba") is not None


if is_numba_available():
    from numba import njit

    @njit(cache=True)
    def _evaluate_mass_action_rates_kernel(
        reactant_orders,
        rate_parameter_indices,
        concentrations,
        parameters,
        clip_negative_concentrations,
    ):
        n_processes = reactant_orders.shape[0]
        n_species = reactant_orders.shape[1]

        rates = np.empty(n_processes, dtype=np.float64)

        for process_index in range(n_processes):
            rate = parameters[rate_parameter_indices[process_index]]

            for species_index in range(n_species):
                order = reactant_orders[process_index, species_index]

                if order != 0.0:
                    concentration = concentrations[species_index]

                    if clip_negative_concentrations and concentration < 0.0:
                        concentration = 0.0

                    rate *= concentration ** order

            rates[process_index] = rate

        return rates

    @njit(cache=True)
    def _evaluate_mass_action_rhs_kernel(
        reactant_orders,
        net_stoichiometry,
        rate_parameter_indices,
        concentrations,
        parameters,
        clip_negative_concentrations,
    ):
        rates = _evaluate_mass_action_rates_kernel(
            reactant_orders,
            rate_parameter_indices,
            concentrations,
            parameters,
            clip_negative_concentrations,
        )

        n_processes = net_stoichiometry.shape[0]
        n_species = net_stoichiometry.shape[1]

        rhs = np.zeros(n_species, dtype=np.float64)

        for process_index in range(n_processes):
            rate = rates[process_index]

            for species_index in range(n_species):
                rhs[species_index] += (
                    rate * net_stoichiometry[process_index, species_index]
                )

        return rhs

else:
    _evaluate_mass_action_rates_kernel = None
    _evaluate_mass_action_rhs_kernel = None


def _require_numba() -> None:
    """
    Raise a clear error if numba is unavailable.
    """

    if not is_numba_available():
        raise RuntimeError(
            "Numba is not installed. Install numba to use the Numba RHS backend."
        )


def evaluate_mass_action_rates_numba(
    compiled_model: CompiledMassActionModel,
    concentrations: np.ndarray,
    parameters: np.ndarray,
    clip_negative_concentrations: bool = False,
) -> np.ndarray:
    """
    Evaluate mass-action process rates using a Numba-compiled kernel.
    """

    _require_numba()

    y = np.asarray(concentrations, dtype=np.float64)
    p = np.asarray(parameters, dtype=np.float64)

    if y.shape != (compiled_model.n_species,):
        raise ValueError(
            f"concentrations must have shape "
            f"({compiled_model.n_species},), got {y.shape}"
        )

    if p.shape != (compiled_model.n_parameters,):
        raise ValueError(
            f"parameters must have shape "
            f"({compiled_model.n_parameters},), got {p.shape}"
        )

    return _evaluate_mass_action_rates_kernel(
        compiled_model.reactant_orders.astype(np.float64),
        compiled_model.rate_parameter_indices.astype(np.int64),
        y,
        p,
        bool(clip_negative_concentrations),
    )


def evaluate_mass_action_rhs_numba(
    compiled_model: CompiledMassActionModel,
    concentrations: np.ndarray,
    parameters: np.ndarray,
    clip_negative_concentrations: bool = False,
) -> np.ndarray:
    """
    Evaluate dy/dt using a Numba-compiled mass-action RHS kernel.
    """

    _require_numba()

    y = np.asarray(concentrations, dtype=np.float64)
    p = np.asarray(parameters, dtype=np.float64)

    if y.shape != (compiled_model.n_species,):
        raise ValueError(
            f"concentrations must have shape "
            f"({compiled_model.n_species},), got {y.shape}"
        )

    if p.shape != (compiled_model.n_parameters,):
        raise ValueError(
            f"parameters must have shape "
            f"({compiled_model.n_parameters},), got {p.shape}"
        )

    return _evaluate_mass_action_rhs_kernel(
        compiled_model.reactant_orders.astype(np.float64),
        compiled_model.net_stoichiometry.astype(np.float64),
        compiled_model.rate_parameter_indices.astype(np.int64),
        y,
        p,
        bool(clip_negative_concentrations),
    )


def warm_up_numba_rhs(
    compiled_model: CompiledMassActionModel,
    concentrations: np.ndarray,
    parameters: np.ndarray,
) -> None:
    """
    Trigger Numba compilation before timing.

    The first Numba call includes compile time, so benchmarks should call this
    before measuring runtime.
    """

    evaluate_mass_action_rhs_numba(
        compiled_model=compiled_model,
        concentrations=concentrations,
        parameters=parameters,
    )
