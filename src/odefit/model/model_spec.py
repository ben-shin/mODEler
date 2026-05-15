from dataclasses import dataclass

from odefit.model.parameters import detect_parameters
from odefit.model.parser import parse_model_text
from odefit.model.reaction import Reaction
from odefit.model.species import detect_species


@dataclass
class ModelSpec:
    """
    Complete representation of a kinetic model

    This object is built from raw reaction text and contains:
    - The original text
    - Parsed reactions
    - Detected species
    - Detected kinetic parameters
    """

    raw_text: str
    reactions: list[Reaction]
    species: list[str]
    parameters: list[str]


def build_model_spec(text: str) -> ModelSpec:
    """
    Build a complete model+species from raw model text
    """

    reactions = parse_model_text(text)
    species = detect_species(reactions)
    parameters = detect_parameters(reactions)

    return ModelSpec(
        raw_text=text,
        reactions=reactions,
        species=species,
        parameters=parameters,
    )
