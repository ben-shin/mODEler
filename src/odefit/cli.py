from __future__ import annotations

import argparse
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


def parse_mapping_entries(
    mapping_entries: list[str] | None,
) -> dict[str, str]:
    """
    Parse mapping entries of the form:

        data_column:species

    Example:
        amide:A
    """

    if not mapping_entries:
        return {}

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


def parse_parameter_entries(
    parameter_entries: list[str] | None,
) -> dict[str, tuple[float, float, float]]:
    """
    Parse parameter entries of the form:

        name:initial_guess:lower_bound:upper_bound

    Example:
        k1f:0.01:0:10
    """

    if not parameter_entries:
        return {}

    parsed: dict[str, tuple[float, float, float]] = {}

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


def parse_initial_condition_entries(
    initial_entries: list[str] | None,
) -> dict[str, tuple[float, bool, float, float]]:
    """
    Parse initial-condition entries of the form:

        species:value:fixed_or_fit:lower_bound:upper_bound

    Examples:
        A:1.0:fixed:0:10
        A:1.0:fit:0:10
    """

    if not initial_entries:
        return {}

    parsed: dict[str, tuple[float, bool, float, float]] = {}

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

        fixed = mode == "fixed"

        parsed[species] = (
            float(value),
            fixed,
            float(lower),
            float(upper),
        )

    return parsed


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


def parse_signal_weight_entries(
    weight_entries: list[str] | None,
) -> dict[str, float] | None:
    """
    Parse signal weight entries of the form:

        data_column:weight

    Example:
        amide:2.0
    """

    if not weight_entries:
        return None

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
    """

    model = read_model_file(args.model)

    dataframe = pd.read_csv(args.data)

    dataset = Dataset(
        raw_dataframe=dataframe,
        time_column=args.time_column,
        signal_columns=args.signal_columns,
    )

    species_mapping = parse_mapping_entries(args.mapping)

    if not species_mapping:
        species_mapping = build_default_species_mapping(
            signal_columns=args.signal_columns,
            model=model,
        )

    parameter_specs = build_parameter_specs(
        model=model,
        parameter_entries=args.parameter,
        default_guess=args.default_parameter_guess,
        default_lower=args.default_parameter_lower,
        default_upper=args.default_parameter_upper,
    )

    initial_condition_specs = build_initial_condition_specs(
        model=model,
        initial_entries=args.initial,
    )

    settings = FitSettings(
        species_mapping=species_mapping,
        use_normalized_data=False,
        method=args.method,
        loss=args.loss,
        max_nfev=args.max_nfev,
        rtol=args.rtol,
        atol=args.atol,
        signal_weights=parse_signal_weight_entries(args.signal_weight),
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
        output_dir=args.output_dir,
        parameter_specs=parameter_specs,
        initial_condition_specs=initial_condition_specs,
        species_mapping=species_mapping,
        include_plots=not args.no_plots,
    )

    print("Fit success:", result.success)
    print("Message:", result.message)
    print("Fitted parameters:", result.fitted_parameters)
    print("Fitted initial conditions:", result.fitted_initial_conditions)
    print("Statistics:", result.statistics)
    print(f"\nWrote output bundle to: {Path(args.output_dir)}")

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
        "--model",
        required=True,
        help="Path to model text file.",
    )

    fit_parser.add_argument(
        "--data",
        required=True,
        help="Path to CSV data file.",
    )

    fit_parser.add_argument(
        "--time-column",
        default="time",
        help="Name of the time column.",
    )

    fit_parser.add_argument(
        "--signal-columns",
        nargs="+",
        required=True,
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
        default=0.1,
        help="Default initial guess for model parameters.",
    )

    fit_parser.add_argument(
        "--default-parameter-lower",
        type=float,
        default=0.0,
        help="Default lower bound for model parameters.",
    )

    fit_parser.add_argument(
        "--default-parameter-upper",
        type=float,
        default=100.0,
        help="Default upper bound for model parameters.",
    )

    fit_parser.add_argument(
        "--method",
        default="trf",
        help="scipy.optimize.least_squares method.",
    )

    fit_parser.add_argument(
        "--loss",
        default="linear",
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
        default=1e-6,
        help="ODE solver relative tolerance.",
    )

    fit_parser.add_argument(
        "--atol",
        type=float,
        default=1e-9,
        help="ODE solver absolute tolerance.",
    )

    fit_parser.add_argument(
        "--output-dir",
        required=True,
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
