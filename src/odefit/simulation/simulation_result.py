from dataclasses import dataclass, field

import numpy as np


@dataclass
class SimulationResult:
    """
    Result of simulating an ODE model.
    """

    timepoints: np.ndarray
    species: list[str]
    values: np.ndarray
    success: bool = True
    message: str = ""
    warnings: list[str] = field(default_factory=list)

    def get_species_values(
        self,
        species_name: str,
    ) -> np.ndarray:
        """
        Return simulated values for one species.
        """

        if species_name not in self.species:
            raise ValueError(f"Species not found in simulation result: {species_name}")

        species_index = self.species.index(species_name)

        return self.values[:, species_index]
