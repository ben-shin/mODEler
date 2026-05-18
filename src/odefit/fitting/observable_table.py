import pandas as pd

from odefit.fitting.observable_spec import ObservableSpec
from odefit.fitting.observable_vector import ObservableParameters


def build_observable_table(
    observable_specs: list[ObservableSpec],
    fitted_observables: ObservableParameters,
) -> pd.DataFrame:
    """
    Build a tidy observable table for GUI display and CSV export.

    Current observable model:

        observed = scale * species + offset
    """

    rows = []

    for observable in observable_specs:
        data_column = observable.data_column

        if data_column not in fitted_observables:
            raise ValueError(
                f"Missing fitted observable for data column: {data_column}"
            )

        fitted_observable = fitted_observables[data_column]

        rows.append(
            {
                "data_column": data_column,
                "species": observable.species,
                "fitted_species": fitted_observable["species"],
                "scale_initial_guess": observable.scale_initial_guess,
                "scale_fitted_value": fitted_observable["scale"],
                "scale_lower_bound": observable.scale_lower_bound,
                "scale_upper_bound": observable.scale_upper_bound,
                "scale_fixed": observable.scale_fixed,
                "scale_fixed_value": observable.scale_fixed_value,
                "offset_initial_guess": observable.offset_initial_guess,
                "offset_fitted_value": fitted_observable["offset"],
                "offset_lower_bound": observable.offset_lower_bound,
                "offset_upper_bound": observable.offset_upper_bound,
                "offset_fixed": observable.offset_fixed,
                "offset_fixed_value": observable.offset_fixed_value,
            }
        )

    return pd.DataFrame(rows)
