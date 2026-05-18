from dataclasses import dataclass, field
from typing import Any

from odefit.model.parameters import get_parameters_from_reactions
from odefit.model.parser import parse_model_text
from odefit.model.reaction import Reaction
from odefit.model.species import get_species_from_reactions


@dataclass
class ModelSpec:
    """
    Full parsed model specification.
    """

    raw_text: str
    reactions: list[Reaction]
    species: list[str]
    parameters: list[str]
    name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def build_model_warnings(
    reactions: list[Reaction],
    species: list[str],
) -> list[str]:
    """
    Build non-fatal model-level warnings.

    These warnings should not stop parsing. They are intended for GUI/docs/export.
    """

    warnings: list[str] = []

    if not reactions:
        warnings.append("Model contains no reactions")
        return warnings

    reactant_species = set()
    product_species = set()

    for reaction in reactions:
        reactant_species.update(reaction.reactants)
        product_species.update(reaction.products)

    source_species = sorted(
        reactant_species - product_species,
        key=lambda species_name: species.index(species_name),
    )

    sink_species = sorted(
        product_species - reactant_species,
        key=lambda species_name: species.index(species_name),
    )

    if source_species:
        warnings.append(
            "Species appear only as reactants and are never produced: "
            + ", ".join(source_species)
        )

    if sink_species:
        warnings.append(
            "Species appear only as products and are never consumed: "
            + ", ".join(sink_species)
        )

    irreversible_count = sum(not reaction.reversible for reaction in reactions)
    reversible_count = sum(reaction.reversible for reaction in reactions)

    if irreversible_count == len(reactions):
        warnings.append("Model contains only irreversible reactions")

    if reversible_count == len(reactions):
        warnings.append("Model contains only reversible reactions")

    return warnings


def build_model_spec(
    text: str,
    name: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> ModelSpec:
    """
    Parse model text and build a ModelSpec.
    """

    reactions = parse_model_text(text)
    species = get_species_from_reactions(reactions)
    parameters = get_parameters_from_reactions(reactions)

    if metadata is None:
        metadata = {}

    warnings = build_model_warnings(
        reactions=reactions,
        species=species,
    )

    return ModelSpec(
        raw_text=text,
        reactions=reactions,
        species=species,
        parameters=parameters,
        name=name,
        metadata=metadata,
        warnings=warnings,
    )
