from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from odefit.model.model_spec import ModelSpec


@dataclass
class CompiledMassActionModel:
    """
    Array-based mass-action model representation.

    This is designed as a bridge toward Numba/Cython/JAX/Julia backends.

    species:
        Species names in array order.

    parameters:
        Parameter names in array order.

    reactant_orders:
        Matrix with shape (n_processes, n_species).
        Each row contains reaction orders for one elementary process.

    net_stoichiometry:
        Matrix with shape (n_processes, n_species).
        Each row contains species changes caused by one elementary process.

    rate_parameter_indices:
        Array with shape (n_processes,).
        Each entry maps a process to its rate constant parameter index.

    process_labels:
        Human-readable labels for each elementary process.
    """

    species: list[str]
    parameters: list[str]
    reactant_orders: np.ndarray
    net_stoichiometry: np.ndarray
    rate_parameter_indices: np.ndarray
    process_labels: list[str]

    @property
    def n_species(self) -> int:
        return len(self.species)

    @property
    def n_parameters(self) -> int:
        return len(self.parameters)

    @property
    def n_processes(self) -> int:
        return int(self.reactant_orders.shape[0])

    @property
    def species_index(self) -> dict[str, int]:
        return {
            species: index
            for index, species in enumerate(self.species)
        }

    @property
    def parameter_index(self) -> dict[str, int]:
        return {
            parameter: index
            for index, parameter in enumerate(self.parameters)
        }


def _stoichiometry_vector(
    species: list[str],
    stoichiometry: dict[str, int],
) -> np.ndarray:
    """
    Convert a species stoichiometry dictionary to an array.
    """

    vector = np.zeros(len(species), dtype=float)

    species_index = {
        species_name: index
        for index, species_name in enumerate(species)
    }

    for species_name, coefficient in stoichiometry.items():
        if species_name not in species_index:
            raise ValueError(f"Unknown species in stoichiometry: {species_name}")

        vector[species_index[species_name]] = float(coefficient)

    return vector


def compile_mass_action_model(
    model: ModelSpec,
) -> CompiledMassActionModel:
    """
    Compile a ModelSpec into array-based mass-action representation.

    Each irreversible reaction produces one elementary process.
    Each reversible reaction produces two elementary processes:
    one forward and one reverse.
    """

    species = list(model.species)
    parameters = list(model.parameters)

    parameter_index = {
        parameter: index
        for index, parameter in enumerate(parameters)
    }

    reactant_order_rows: list[np.ndarray] = []
    net_stoichiometry_rows: list[np.ndarray] = []
    rate_parameter_indices: list[int] = []
    process_labels: list[str] = []

    for reaction_index, reaction in enumerate(model.reactions, start=1):
        reactants = _stoichiometry_vector(
            species=species,
            stoichiometry=reaction.reactants,
        )

        products = _stoichiometry_vector(
            species=species,
            stoichiometry=reaction.products,
        )

        if reaction.forward_rate not in parameter_index:
            raise ValueError(
                f"Forward rate parameter not found in model parameters: "
                f"{reaction.forward_rate}"
            )

        reactant_order_rows.append(reactants)
        net_stoichiometry_rows.append(products - reactants)
        rate_parameter_indices.append(parameter_index[reaction.forward_rate])

        label = reaction.label

        if label is None:
            label = f"reaction_{reaction_index}"

        process_labels.append(f"{label}_forward")

        if reaction.reversible:
            if reaction.reverse_rate is None:
                raise ValueError(
                    "Reversible reaction is missing reverse rate parameter"
                )

            if reaction.reverse_rate not in parameter_index:
                raise ValueError(
                    f"Reverse rate parameter not found in model parameters: "
                    f"{reaction.reverse_rate}"
                )

            reactant_order_rows.append(products)
            net_stoichiometry_rows.append(reactants - products)
            rate_parameter_indices.append(parameter_index[reaction.reverse_rate])
            process_labels.append(f"{label}_reverse")

    if not reactant_order_rows:
        raise ValueError("Model has no reactions to compile")

    return CompiledMassActionModel(
        species=species,
        parameters=parameters,
        reactant_orders=np.vstack(reactant_order_rows),
        net_stoichiometry=np.vstack(net_stoichiometry_rows),
        rate_parameter_indices=np.asarray(rate_parameter_indices, dtype=int),
        process_labels=process_labels,
    )


def evaluate_mass_action_rates(
    compiled_model: CompiledMassActionModel,
    concentrations: np.ndarray,
    parameters: np.ndarray,
    clip_negative_concentrations: bool = False,
) -> np.ndarray:
    """
    Evaluate mass-action process rates.

    Rate law for each process:

        rate = k * product(concentration_i ** order_i)
    """

    y = np.asarray(concentrations, dtype=float)
    p = np.asarray(parameters, dtype=float)

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

    if clip_negative_concentrations:
        y = np.maximum(y, 0.0)

    rates = np.empty(compiled_model.n_processes, dtype=float)

    for process_index in range(compiled_model.n_processes):
        rate = p[compiled_model.rate_parameter_indices[process_index]]

        for species_index in range(compiled_model.n_species):
            order = compiled_model.reactant_orders[
                process_index,
                species_index,
            ]

            if order != 0.0:
                rate *= y[species_index] ** order

        rates[process_index] = rate

    return rates


def evaluate_mass_action_rhs(
    compiled_model: CompiledMassActionModel,
    concentrations: np.ndarray,
    parameters: np.ndarray,
    clip_negative_concentrations: bool = False,
) -> np.ndarray:
    """
    Evaluate dy/dt for an array-compiled mass-action model.
    """

    rates = evaluate_mass_action_rates(
        compiled_model=compiled_model,
        concentrations=concentrations,
        parameters=parameters,
        clip_negative_concentrations=clip_negative_concentrations,
    )

    return rates @ compiled_model.net_stoichiometry


def parameter_dict_to_array(
    compiled_model: CompiledMassActionModel,
    parameter_values: dict[str, float],
) -> np.ndarray:
    """
    Convert parameter dictionary to array ordered for a compiled model.
    """

    missing = [
        parameter
        for parameter in compiled_model.parameters
        if parameter not in parameter_values
    ]

    if missing:
        raise ValueError(f"Missing parameter values: {', '.join(missing)}")

    return np.asarray(
        [
            float(parameter_values[parameter])
            for parameter in compiled_model.parameters
        ],
        dtype=float,
    )


def concentration_dict_to_array(
    compiled_model: CompiledMassActionModel,
    concentrations: dict[str, float],
) -> np.ndarray:
    """
    Convert species concentration dictionary to array ordered for a compiled model.
    """

    missing = [
        species
        for species in compiled_model.species
        if species not in concentrations
    ]

    if missing:
        raise ValueError(f"Missing concentration values: {', '.join(missing)}")

    return np.asarray(
        [
            float(concentrations[species])
            for species in compiled_model.species
        ],
        dtype=float,
    )


def array_to_species_dict(
    compiled_model: CompiledMassActionModel,
    values: np.ndarray,
) -> dict[str, float]:
    """
    Convert species array to dictionary.
    """

    values = np.asarray(values, dtype=float)

    if values.shape != (compiled_model.n_species,):
        raise ValueError(
            f"values must have shape ({compiled_model.n_species},), "
            f"got {values.shape}"
        )

    return {
        species: float(values[index])
        for index, species in enumerate(compiled_model.species)
    }
