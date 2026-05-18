from pathlib import Path

from odefit.model.model_spec import ModelSpec, build_model_spec
from odefit.model.ode_generator import generate_ode_lines
from odefit.model.reaction import Reaction


def build_model_spec_from_file(
    file_path: str | Path,
    name: str | None = None,
    metadata: dict | None = None,
) -> ModelSpec:
    """
    Read a model text file and build a ModelSpec.
    """

    path = Path(file_path)

    text = path.read_text()

    if metadata is None:
        metadata = {}

    metadata = {
        **metadata,
        "source_file": str(path),
    }

    return build_model_spec(
        text=text,
        name=name,
        metadata=metadata,
    )


def write_generated_odes(
    model: ModelSpec,
    file_path: str | Path,
) -> Path:
    """
    Write generated ODE lines to a text file.
    """

    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ode_lines = generate_ode_lines(model)

    path.write_text("\n".join(ode_lines) + "\n")

    return path


def format_reaction_side(
    species_counts: dict[str, int],
) -> str:
    """
    Format a reaction side from a species-count dictionary.

    Examples:
        {"A": 1} -> A
        {"A": 2} -> 2A
        {"A": 1, "B": 1} -> A + B
    """

    terms: list[str] = []

    for species, coefficient in species_counts.items():
        if coefficient == 1:
            terms.append(species)
        else:
            terms.append(f"{coefficient}{species}")

    return " + ".join(terms)


def format_reaction(
    reaction: Reaction,
) -> str:
    """
    Format one Reaction as readable reaction text.
    """

    left_side = format_reaction_side(reaction.reactants)
    right_side = format_reaction_side(reaction.products)

    if reaction.reversible:
        arrow = "<->"
    else:
        arrow = "->"

    reaction_text = f"{left_side} {arrow} {right_side}"

    if reaction.label is not None:
        reaction_text = f"{reaction.label}: {reaction_text}"

    return reaction_text


def build_reaction_table_text(
    model: ModelSpec,
) -> str:
    """
    Build a readable reaction table.

    This is useful for text export, reports, and GUI previews.
    """

    lines: list[str] = []

    lines.append("Reaction table")
    lines.append("==============")
    lines.append("")

    if not model.reactions:
        lines.append("No reactions.")
        return "\n".join(lines) + "\n"

    header = [
        "index",
        "source_line",
        "label",
        "reaction",
        "reversible",
        "forward_rate",
        "reverse_rate",
    ]

    lines.append("\t".join(header))

    for index, reaction in enumerate(model.reactions, start=1):
        row = [
            str(index),
            "" if reaction.source_line is None else str(reaction.source_line),
            "" if reaction.label is None else reaction.label,
            format_reaction(reaction),
            str(reaction.reversible),
            reaction.forward_rate,
            "" if reaction.reverse_rate is None else reaction.reverse_rate,
        ]

        lines.append("\t".join(row))

    return "\n".join(lines) + "\n"


def build_model_summary_text(
    model: ModelSpec,
) -> str:
    """
    Build a readable model summary.

    Includes:
    - model name
    - metadata
    - species
    - parameters
    - warnings
    - generated ODEs
    - reaction table
    """

    lines: list[str] = []

    lines.append("Model summary")
    lines.append("=============")
    lines.append("")

    if model.name is not None:
        lines.append(f"Name: {model.name}")
    else:
        lines.append("Name: unnamed model")

    lines.append("")

    lines.append("Metadata")
    lines.append("--------")

    if model.metadata:
        for key, value in model.metadata.items():
            lines.append(f"{key}: {value}")
    else:
        lines.append("No metadata.")

    lines.append("")

    lines.append("Species")
    lines.append("-------")

    if model.species:
        for species in model.species:
            lines.append(f"- {species}")
    else:
        lines.append("No species detected.")

    lines.append("")

    lines.append("Parameters")
    lines.append("----------")

    if model.parameters:
        for parameter in model.parameters:
            lines.append(f"- {parameter}")
    else:
        lines.append("No parameters detected.")

    lines.append("")

    lines.append("Warnings")
    lines.append("--------")

    if model.warnings:
        for warning in model.warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("No warnings.")

    lines.append("")

    lines.append("Generated ODEs")
    lines.append("--------------")

    ode_lines = generate_ode_lines(model)

    if ode_lines:
        for ode_line in ode_lines:
            lines.append(ode_line)
    else:
        lines.append("No ODEs generated.")

    lines.append("")
    lines.append(build_reaction_table_text(model).rstrip())

    return "\n".join(lines) + "\n"


def write_model_summary(
    model: ModelSpec,
    file_path: str | Path,
) -> Path:
    """
    Write model summary text to a file.
    """

    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(build_model_summary_text(model))

    return path


def write_reaction_table(
    model: ModelSpec,
    file_path: str | Path,
) -> Path:
    """
    Write reaction table text to a file.
    """

    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(build_reaction_table_text(model))

    return path
