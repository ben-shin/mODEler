import pandas as pd

from odefit.fitting.parameter_spec import ParameterSpec


def build_parameter_table(
    parameter_specs: list[ParameterSpec],
    fitted_parameters: dict[str, float],
) -> pd.DataFrame:
    """
    Build a tidy parameter table for GUI display and CSV export.
    """

    rows = []

    for parameter in parameter_specs:
        if parameter.name not in fitted_parameters:
            raise ValueError(f"Missing fitted value for parameter: {parameter.name}")

        rows.append(
            {
                "parameter": parameter.name,
                "initial_guess": parameter.initial_guess,
                "fitted_value": fitted_parameters[parameter.name],
                "lower_bound": parameter.lower_bound,
                "upper_bound": parameter.upper_bound,
                "fixed": parameter.fixed,
                "fixed_value": parameter.fixed_value,
            }
        )

    return pd.DataFrame(rows)


def build_initial_parameter_dict(
    parameter_specs: list[ParameterSpec],
) -> dict[str, float]:
    """
    Build a dictionary of initial/fixed parameter values.

    Fixed parameters use fixed_value.
    Free parameters use initial_guess.
    """

    initial_parameters: dict[str, float] = {}

    for parameter in parameter_specs:
        if parameter.fixed:
            if parameter.fixed_value is None:
                raise ValueError(
                    f"Fixed parameter missing fixed value: {parameter.name}"
                )

            initial_parameters[parameter.name] = parameter.fixed_value

        else:
            initial_parameters[parameter.name] = parameter.initial_guess

    return initial_parameters
