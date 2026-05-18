import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from odefit.data.dataset import Dataset
from odefit.fitting.fit_settings import FitSettings
from odefit.fitting.initial_condition_spec import InitialConditionSpec
from odefit.fitting.optimizer import fit_model
from odefit.fitting.parameter_spec import ParameterSpec
from odefit.model.model_spec import build_model_spec


def fit_amide_dimerization(
    data_path: str,
    time_column: str = "time",
    signal_column: str = "amide",
    output_plot: str | None = None,
):
    """
    Fit integrated amide-region NMR signal using:

        2A ⇌ A2

    Assumption:
        observed amide signal is proportional to monomer/NMR-visible A.
    """

    dataframe = pd.read_csv(data_path)

    dataset = Dataset(
        raw_dataframe=dataframe,
        time_column=time_column,
        signal_columns=[signal_column],
    )

    model = build_model_spec("2A-A2")

    print("Detected species:")
    print(model.species)

    print("\nDetected parameters:")
    print(model.parameters)

    # For 2A-A2:
    # k1f = forward dimerization rate
    # k1r = reverse dissociation rate
    parameter_specs = [
        ParameterSpec(
            name="k1f",
            initial_guess=0.01,
            lower_bound=0.0,
            upper_bound=100.0,
        ),
        ParameterSpec(
            name="k1r",
            initial_guess=0.01,
            lower_bound=0.0,
            upper_bound=100.0,
        ),
    ]

    first_signal_value = float(dataframe[signal_column].iloc[0])

    initial_condition_specs = [
        InitialConditionSpec(
            species="A",
            initial_guess=first_signal_value,
            lower_bound=0.0,
            upper_bound=first_signal_value * 10.0,
            fixed=False,
        ),
        InitialConditionSpec(
            species="A2",
            initial_guess=0.0,
            lower_bound=0.0,
            upper_bound=first_signal_value * 10.0,
            fixed=True,
            fixed_value=0.0,
        ),
    ]

    settings = FitSettings(
        species_mapping={
            signal_column: "A",
        },
        rtol=1e-8,
        atol=1e-10,
    )

    result = fit_model(
        model=model,
        dataset=dataset,
        parameter_specs=parameter_specs,
        initial_condition_specs=initial_condition_specs,
        settings=settings,
    )

    print("\nFit success:")
    print(result.success)

    print("\nOptimizer message:")
    print(result.message)

    print("\nFitted kinetic parameters:")
    for name, value in result.fitted_parameters.items():
        print(f"{name}: {value}")

    print("\nFitted initial conditions:")
    for name, value in result.fitted_initial_conditions.items():
        print(f"{name}0: {value}")

    print("\nFit statistics:")
    for name, value in result.statistics.items():
        print(f"{name}: {value}")

    time_values = dataframe[time_column]
    observed_signal = dataframe[signal_column]

    fitted_a = result.simulation_result.get_species_values("A")
    fitted_a2 = result.simulation_result.get_species_values("A2")

    plt.figure()
    plt.scatter(
        time_values,
        observed_signal,
        label=f"Observed {signal_column}",
    )
    plt.plot(
        result.simulation_result.timepoints,
        fitted_a,
        label="Fitted A",
    )
    plt.xlabel(time_column)
    plt.ylabel(signal_column)
    plt.legend()
    plt.tight_layout()

    if output_plot is not None:
        output_path = Path(output_plot)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300)
        print(f"\nSaved plot to: {output_path}")

    plt.show()

    plt.figure()
    plt.plot(
        result.simulation_result.timepoints,
        fitted_a,
        label="A",
    )
    plt.plot(
        result.simulation_result.timepoints,
        fitted_a2,
        label="A2",
    )
    plt.xlabel(time_column)
    plt.ylabel("Fitted concentration / signal units")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Fit integrated amide-region time course using 2A ⇌ A2."
    )

    parser.add_argument(
        "data_path",
        help="Path to CSV file containing time-course data.",
    )

    parser.add_argument(
        "--time-column",
        default="time",
        help="Name of the time column. Default: time",
    )

    parser.add_argument(
        "--signal-column",
        default="amide",
        help="Name of the integrated amide signal column. Default: amide",
    )

    parser.add_argument(
        "--output-plot",
        default=None,
        help="Optional path to save observed-vs-fit plot.",
    )

    args = parser.parse_args()

    fit_amide_dimerization(
        data_path=args.data_path,
        time_column=args.time_column,
        signal_column=args.signal_column,
        output_plot=args.output_plot,
    )


if __name__ == "__main__":
    main()
