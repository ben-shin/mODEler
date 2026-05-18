from dataclasses import dataclass


@dataclass
class Reaction:
    reactants: dict[str, int]
    products: dict[str, int]
    reversible: bool
    forward_rate: str
    reverse_rate: str | None = None
    label: str | None = None
    source_line: int | None = None
