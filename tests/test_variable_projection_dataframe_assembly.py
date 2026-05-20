import warnings

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning

from odefit.engines.registry import get_engine_bundle
from odefit.fitting.variable_projection import project_observables_onto_species


def test_project_observables_many_columns_avoids_dataframe_fragmentation_warning():
    timepoints = np.linspace(0.0, 10.0, 30)
    species_values = np.exp(-0.4 * timepoints)

    data = {"time": timepoints}
    signal_columns = []

    for index in range(150):
        column = f"peak_{index}"
        signal_columns.append(column)
        data[column] = (1.0 + index * 0.001) * species_values + 0.01

    dataframe = pd.DataFrame(data)
    engine_bundle = get_engine_bundle("reference")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        result = project_observables_onto_species(
            timepoints=timepoints,
            simulated_species_values=species_values,
            observed_dataframe=dataframe,
            signal_columns=signal_columns,
            fit_scale=True,
            fit_offset=True,
            engine_bundle=engine_bundle,
        )

    performance_warnings = [
        warning
        for warning in caught
        if issubclass(warning.category, PerformanceWarning)
    ]

    assert not performance_warnings
    assert result.observable_table.shape[0] == len(signal_columns)
    assert result.predicted_dataframe.shape == dataframe.shape
    assert result.residuals_dataframe.shape == dataframe.shape
    assert result.rss >= 0.0
