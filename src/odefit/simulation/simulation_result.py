from dataclasses import dataclass

import numpy as np


@dataclass
class SimulationResult:
    """
    Result of an ODE simulation.

    values has shape:
        number of timepoints x number of species
    """

    timepoints: np.ndarray
    species: list[str]
    values: np.ndarray

    def get_species_values(self, species_name: str) -> np.ndarray:
        """
        Return simulated values for one species.
        """

        species_index = self.species.index(species_name)

        return self.values[:, species_index]
