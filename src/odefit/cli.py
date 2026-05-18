from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from odefit.data.dataset import Dataset
from odefit.export.bundle_export import export_fit_bundle
from odefit.fitting.fit_settings import FitSettings
from odefit.fitting.initial_condition_spec import InitialConditionSpec
from odefit.fitting.optimizer import fit_model
from odefit.fitting.parameter_spec import ParameterSpec
from odefit.model.model_spec import ModelSpec, build_model_spec
from odefit.model.ode_generator import generate_ode_lines


def read_model_file(model_path: str | Path) -> ModelSpec:
    """
    Read a model text file and build a ModelSpec.
    """

    path = Path(model_path)

    if not path.exists():
        raise FileNotFoundError(f"Model file does not exist: {path}")

    return build_model_spec(path.read_text())


def load_fit_config(config_path: str | Path | None) -> dict:
    """
    Load a JSON fit configuration file.

    Returns an empty dict if no config path is supplied.
    """

    if config_path is None:
        return {}

    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")

    with path.open("r") as handle:
        return json.load(handle)


def get_config_value(
    args: argparse.Namespace,
    config: dict,
    argument_name: str,
    default=None,
    required: bool = False,
):
    """
    Get an argument value, preferring command-line value over config value.
    """

    cli_value = getattr(args, argument_name)

    if cli_value is not None:
        value = cli_value
    else:
        value = config.get(argument_name, default)

    if required and value is None:
        raise ValueError(f"Missing required argument/config value: {argument_name}")

    return value


def get_config_list_or_dict_value(
    args: argparse.Namespace,
    config: dict,
    argument_name: str,
    alternative_config_name: str | None = None,
):
    """
    Get a list/dict config value, preferring CLI over config.

    Supports alternative config keys like:
        parameter -> parameters
        initial -> initial_conditions
        signal_weight -> signal_weights
    """

    cli_value = getattr(args, argument_name)

    if cli_value is not None:
        return cli_value

    if argument_name in config:
        return config[argument_name]

    if alternative_config_name is not None and alternative_config_name in config:
        return config[alternative_config_name]

    return None


def parse_mapping_entries(
    mapping_entries: list[str] | dict[str, str] | None,
) -> dict[str, str]:
    """
    Parse mapping entries.

    CLI format:
        ["data_column:species"]

    JSON config format:
        {
            "data_column": "species"
        }
    """

    if not mapping_entries:
        return {}

    if isinstance(mapping_entries, dict):
        return {
            str(data_column): str(species)
            for data_column, species in mapping_entries.items()
        }

    mapping: dict[str, str] = {}

    for entry in mapping_entries:
        parts = entry.split(":")

        if len(parts) != 2:
            raise ValueError(
                f"Mapping entries must have format data_column:species. Got: {entry}"
            )

        data_column, species = parts

        if not data_column or not species:
            raise ValueError(f"Invalid mapping entry: {entry}")

        mapping[data_column] = species

    return mapping


def parse_parameter_entries(
    parameter_entries: list[str] | dict | None,
) -> dict[str, tuple[float, float, float]]:
    """
    Parse parameter entries.

    CLI format:
        ["k1f:0.01:0:10"]

    JSON config format:
        {
            "k1f": {
                "initial_guess": 0.01,
                "lower_bound": 0.0,
                "upper_bound": 10.0
            }
        }
    """

    if not parameter_entries:
        return {}

    parsed: dict[str, tuple[float, float, float]] = {}

    if isinstance(parameter_entries, dict):
        for name, values in parameter_entries.items():
            parsed[str(name)] = (
                float(values["initial_guess"]),
                float(values.get("lower_bound", 0.0)),
                float(values.get("upper_bound", 100.0)),
            )

        return parsed

    for entry in parameter_entries:
        parts = entry.split(":")

        if len(parts) != 4:
            raise ValueError(
                "Parameter entries must have format "
                "name:initial_guess:lower_bound:upper_bound. "
                f"Got: {entry}"
            )

        name, guess, lower, upper = parts

        parsed[name] = (
            float(guess),
            float(lower),
            float(upper),
        )

    return parsed


def parse_initial_condition_entries(
    initial_entries: list[str] | dict | None,
) -> dict[str, tuple[float, bool, float, float]]:
    """
    Parse initial-condition entries.

    CLI format:
        ["A:1.0:fixed:0:10"]

    JSON config format:
        {
            "A": {
                "value": 1.0,
                "mode": "fixed",
                "lower_bound": 0.0,
                "upper_bound": 10.0
            }
        }
    """

    if not initial_entries:
        return {}

    parsed: dict[str, tuple[float, bool, float, float]] = {}

    if isinstance(initial_entries, dict):
        for species, values in initial_entries.items():
            mode = values.get("mode", "fixed")

            if mode not in {"fixed", "fit"}:
                raise ValueError(
                    f"Initial condition mode must be 'fixed' or 'fit'. Got: {mode}"
                )

            parsed[str(species)] = (
                float(values["value"]),
                mode == "fixed",
                float(values.get("lower_bound", 0.0)),
                float(values.get("upper_bound", 100.0)),
            )

        return parsed

    for entry in initial_entries:
        parts = entry.split(":")

        if len(parts) != 5:
            raise ValueError(
                "Initial condition entries must have format "
                "species:value:fixed_or_fit:lower_bound:upper_bound. "
                f"Got: {entry}"
            )

        species, value, mode, lower, upper = parts

        if mode not in {"fixed", "fit"}:
            raise ValueError(
                f"Initial condition mode must be 'fixed' or 'fit'. Got: {mode}"
            )

        parsed[species] = (
            float(value),
            mode == "fixed",
            float(lower),
            float(upper),
        )

    return parsed


def parse_signal_weight_entries(
    weight_entries: list[str] | dict[str, float] | None,
) -> dict[str, float] | None:
    """
    Parse signal weight entries.

    CLI format:
        ["amide:2.0"]

    JSON config format:
        {
            "amide": 2.0
        }
    """

    if not weight_entries:
        return None

    if isinstance(weight_entries, dict):
        return {
            str(data_column): float(weight)
            for data_column, weight in weight_entries.items()
        }

    weights: dict[str, float] = {}

    for entry in weight_entries:
        parts = entry.split(":")

        if len(parts) != 2:
            raise ValueError(
                "Signal weight entries must have format data_column:weight. "
                f"Got: {entry}"
            )

        data_column, weight = parts

        weights[data_column] = float(weight)

    return weights


def build_default_species_mapping(
    signal_columns: list[str],
    model: ModelSpec,
) -> dict[str, str]:
    """
    Build a default data-column to species mapping.

    Rules:
    - If there is one signal column and species A exists, map signal -> A.
    - Otherwise, map columns to species with identical names.
    """

    if len(signal_columns) == 1 and "A" in model.species:
        return {
            signal_columns[0]: "A",
        }

    mapping = {}

    for signal_column in signal_columns:
        if signal_column in model.species:
            mapping[signal_column] = signal_column

    missing_columns = set(signal_columns) - set(mapping)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(
            "Could not infer mapping for signal columns: "
            f"{missing}. Use --mapping data_column:species."
        )

    return mapping


def build_parameter_specs(
    model: ModelSpec,
    parameter_entries: list[str] | None = None,
    default_guess: float = 0.1,
    default_lower: float = 0.0,
    default_upper: float = 100.0,
) -> list[ParameterSpec]:
    """
    Build ParameterSpec objects for model parameters.

    CLI parameter entries override defaults.
    """

    overrides = parse_parameter_entries(parameter_entries)

    unknown_parameters = set(overrides) - set(model.parameters)

    if unknown_parameters:
        unknown = ", ".join(sorted(unknown_parameters))
        raise ValueError(f"Parameter override not present in model: {unknown}")

    parameter_specs = []

    for parameter_name in model.parameters:
        if parameter_name in overrides:
            guess, lower, upper = overrides[parameter_name]
        else:
            guess = default_guess
            lower = default_lower
            upper = default_upper

        parameter_specs.append(
            ParameterSpec(
                name=parameter_name,
                initial_guess=guess,
                lower_bound=lower,
                upper_bound=upper,
            )
        )

    return parameter_specs


def build_initial_condition_specs(
    model: ModelSpec,
    initial_entries: list[str] | None = None,
) -> list[InitialConditionSpec]:
    """
    Build InitialConditionSpec objects.

    Defaults:
    - A starts fixed at 1.0 if A exists.
    - Otherwise, the first species starts fixed at 1.0.
    - Other species start fixed at 0.0.
    """

    overrides = parse_initial_condition_entries(initial_entries)

    unknown_species = set(overrides) - set(model.species)

    if unknown_species:
        unknown = ", ".join(sorted(unknown_species))
        raise ValueError(f"Initial condition species not present in model: {unknown}")

    primary_species = "A" if "A" in model.species else model.species[0]

    initial_condition_specs = []

    for species_name in model.species:
        if species_name in overrides:
            value, fixed, lower, upper = overrides[species_name]
        else:
            value = 1.0 if species_name == primary_species else 0.0
            fixed = True
            lower = 0.0
            upper = 100.0

        initial_condition_specs.append(
            InitialConditionSpec(
                species=species_name,
                initial_guess=value,
                lower_bound=lower,
                upper_bound=upper,
                fixed=fixed,
                fixed_value=value if fixed else None,
            )
        )

    return initial_condition_specs


def command_generate_odes(args: argparse.Namespace) -> None:
    """
    Generate ODE text from a model file.
    """

    model = read_model_file(args.model)

    ode_lines = generate_ode_lines(model)

    print("Detected species:")
    for species in model.species:
        print(f"  {species}")

    print("\nDetected parameters:")
    for parameter in model.parameters:
        print(f"  {parameter}")

    print("\nGenerated ODEs:")
    for line in ode_lines:
        print(line)

    if args.output is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(ode_lines) + "\n")
        print(f"\nWrote ODEs to: {output_path}")


def command_fit(args: argparse.Namespace) -> None:
    """
    Fit a model to a CSV dataset using direct species mapping.

    Values can come from CLI arguments or from a JSON config file.
    CLI arguments override config-file values.
    """

    config = load_fit_config(args.config)

    model_path = get_config_value(
        args=args,
        config=config,
        argument_name="model",
        required=True,
    )

    data_path = get_config_value(
        args=args,
        config=config,
        argument_name="data",
        required=True,
    )

    time_column = get_config_value(
        args=args,
        config=config,
        argument_name="time_column",
        default="time",
    )

    signal_columns = get_config_value(
        args=args,
        config=config,
        argument_name="signal_columns",
        required=True,
    )

    output_dir = get_config_value(
        args=args,
        config=config,
        argument_name="output_dir",
        required=True,
    )

    model = read_model_file(model_path)

    dataframe = pd.read_csv(data_path)

    dataset = Dataset(
        raw_dataframe=dataframe,
        time_column=time_column,
        signal_columns=signal_columns,
    )

    mapping_entries = get_config_list_or_dict_value(
        args=args,
        config=config,
        argument_name="mapping",
    )

    species_mapping = parse_mapping_entries(mapping_entries)

    if not species_mapping:
        species_mapping = build_default_species_mapping(
            signal_columns=signal_columns,
            model=model,
        )

    parameter_entries = get_config_list_or_dict_value(
        args=args,
        config=config,
        argument_name="parameter",
        alternative_config_name="parameters",
    )

    initial_entries = get_config_list_or_dict_value(
        args=args,
        config=config,
        argument_name="initial",
        alternative_config_name="initial_conditions",
    )

    signal_weight_entries = get_config_list_or_dict_value(
        args=args,
        config=config,
        argument_name="signal_weight",
        alternative_config_name="signal_weights",
    )

    default_parameter_guess = get_config_value(
        args=args,
        config=config,
        argument_name="default_parameter_guess",
        default=0.1,
    )

    default_parameter_lower = get_config_value(
        args=args,
        config=config,
        argument_name="default_parameter_lower",
        default=0.0,
    )

    default_parameter_upper = get_config_value(
        args=args,
        config=config,
        argument_name="default_parameter_upper",
        default=100.0,
    )

    method = get_config_value(
        args=args,
        config=config,
        argument_name="method",
        default="trf",
    )

    loss = get_config_value(
        args=args,
        config=config,
        argument_name="loss",
        default="linear",
    )

    max_nfev = get_config_value(
        args=args,
        config=config,
        argument_name="max_nfev",
        default=None,
    )

    rtol = get_config_value(
        args=args,
        config=config,
        argument_name="rtol",
        default=1e-6,
    )

    atol = get_config_value(
        args=args,
        config=config,
        argument_name="atol",
        default=1e-9,
    )

    no_plots = bool(config.get("no_plots", False)) or bool(args.no_plots)

    parameter_specs = build_parameter_specs(
        model=model,
        parameter_entries=parameter_entries,
        default_guess=default_parameter_guess,
        default_lower=default_parameter_lower,
        default_upper=default_parameter_upper,
    )

    initial_condition_specs = build_initial_condition_specs(
        model=model,
        initial_entries=initial_entries,
    )

    settings = FitSettings(
        species_mapping=species_mapping,
        use_normalized_data=False,
        method=method,
        loss=loss,
        max_nfev=max_nfev,
        rtol=rtol,
        atol=atol,
        signal_weights=parse_signal_weight_entries(signal_weight_entries),
    )

    result = fit_model(
        model=model,
        dataset=dataset,
        parameter_specs=parameter_specs,
        initial_condition_specs=initial_condition_specs,
        settings=settings,
    )

    written_files = export_fit_bundle(
        fit_result=result,
        model=model,
        dataset=dataset,
        output_dir=output_dir,
        parameter_specs=parameter_specs,
        initial_condition_specs=initial_condition_specs,
        species_mapping=species_mapping,
        include_plots=not no_plots,
    )

    print("Fit success:", result.success)
    print("Message:", result.message)
    print("Fitted parameters:", result.fitted_parameters)
    print("Fitted initial conditions:", result.fitted_initial_conditions)
    print("Statistics:", result.statistics)
    print(f"\nWrote output bundle to: {Path(output_dir)}")

    print("\nWritten files:")
    for name, path in written_files.items():
        print(f"  {name}: {path}")


def build_parser() -> argparse.ArgumentParser:
    """
    Build CLI argument parser.
    """

    parser = argparse.ArgumentParser(
        prog="odefit",
        description="Build, simulate, fit, and export ODE models.",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    generate_parser = subparsers.add_parser(
        "generate-odes",
        help="Generate ODEs from a reaction model text file.",
    )

    generate_parser.add_argument(
        "--model",
        required=True,
        help="Path to model text file.",
    )

    generate_parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write generated ODEs.",
    )

    generate_parser.set_defaults(func=command_generate_odes)

    fit_parser = subparsers.add_parser(
        "fit",
        help="Fit a reaction model to a CSV dataset.",
    )

    fit_parser.add_argument(
        "--config",
        default=None,
        help="Path to JSON fit configuration file.",
    )

    fit_parser.add_argument(
        "--model",
        default=None,
        help="Path to model text file.",
    )

    fit_parser.add_argument(
        "--data",
        default=None,
        help="Path to CSV data file.",
    )

    fit_parser.add_argument(
        "--time-column",
        default=None,
        help="Name of the time column.",
    )

    fit_parser.add_argument(
        "--signal-columns",
        nargs="+",
        default=None,
        help="Signal/data columns to fit.",
    )

    fit_parser.add_argument(
        "--mapping",
        action="append",
        default=None,
        help="Data-column to species mapping: data_column:species. Can be repeated.",
    )

    fit_parser.add_argument(
        "--parameter",
        action="append",
        default=None,
        help=(
            "Parameter override: name:initial_guess:lower_bound:upper_bound. "
            "Can be repeated."
        ),
    )

    fit_parser.add_argument(
        "--initial",
        action="append",
        default=None,
        help=(
            "Initial condition: species:value:fixed_or_fit:lower_bound:upper_bound. "
            "Example: A:1.0:fixed:0:10. Can be repeated."
        ),
    )

    fit_parser.add_argument(
        "--signal-weight",
        action="append",
        default=None,
        help="Signal residual weight: data_column:weight. Can be repeated.",
    )

    fit_parser.add_argument(
        "--default-parameter-guess",
        type=float,
        default=None,
        help="Default initial guess for model parameters.",
    )

    fit_parser.add_argument(
        "--default-parameter-lower",
        type=float,
        default=None,
        help="Default lower bound for model parameters.",
    )

    fit_parser.add_argument(
        "--default-parameter-upper",
        type=float,
        default=None,
        help="Default upper bound for model parameters.",
    )

    fit_parser.add_argument(
        "--method",
        default=None,
        help="scipy.optimize.least_squares method.",
    )

    fit_parser.add_argument(
        "--loss",
        default=None,
        help="scipy.optimize.least_squares loss.",
    )

    fit_parser.add_argument(
        "--max-nfev",
        type=int,
        default=None,
        help="Maximum number of function evaluations.",
    )

    fit_parser.add_argument(
        "--rtol",
        type=float,
        default=None,
        help="ODE solver relative tolerance.",
    )

    fit_parser.add_argument(
        "--atol",
        type=float,
        default=None,
        help="ODE solver absolute tolerance.",
    )

    fit_parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for fit bundle.",
    )

    fit_parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation.",
    )

    fit_parser.set_defaults(func=command_fit)

    return parser


def main(argv: list[str] | None = None) -> None:
    """
    CLI entry point.
    """

    parser = build_parser()
    args = parser.parse_args(argv)

    args.func(args)


if __name__ == "__main__":
    main()
