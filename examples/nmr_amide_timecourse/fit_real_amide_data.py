from pathlib import Path

import pandas as pd

from odefit.data.dataset import Dataset
from odefit.export.bundle_export import export_fit_bundle
from odefit.fitting.fit_settings import FitSettings
from odefit.fitting.initial_condition_spec import InitialConditionSpec
from odefit.fitting.model_comparison import build_ranked_model_comparison_table
from odefit.fitting.observable_spec import ObservableSpec
from odefit.fitting.optimizer import fit_model
from odefit.fitting.parameter_spec import ParameterSpec
from odefit.model.model_spec import build_model_spec


EXAMPLE_DIR = Path(__file__).parent
OUTPUT_DIR = EXAMPLE_DIR / "real_outputs"

DATA_PATH = EXAMPLE_DIR / "example_data.csv"


MODELS = {
    "first_order_loss": "A>B",
    "reversible_conversion": "A-B",
    "irreversible_dimer": "2A>A2",
    "reversible_dimer": "2A-A2",
}


def read_real_amide_dataset(data_path: Path) -> Dataset:
    """
    Read real NMR amide integral timecourse data.

    Uses:
        Elapsed_Hours as time
        Normalized_Integral (%) as signal

    Renames columns internally to simpler names.
    """

    dataframe = pd.read_csv(data_path)

    dataframe = dataframe.rename(
        columns={
            "Elapsed_Hours": "time",
            "Normalized_Integral (%)": "amide_percent",
            "Integral": "amide_integral",
        }
    )

    return Dataset(
        raw_dataframe=dataframe,
        time_column="time",
        signal_columns=["amide_percent"],
    )


def make_parameter_specs(model) -> list[ParameterSpec]:
    """
    Create default kinetic parameter guesses for each model.

    Rate constants are constrained to be positive.
    """

    parameter_specs = []

    for parameter_name in model.parameters:
        parameter_specs.append(
            ParameterSpec(
                name=parameter_name,
                initial_guess=0.01,
                lower_bound=0.0,
                upper_bound=10.0,
            )
        )

    return parameter_specs


def make_initial_condition_specs(model) -> list[InitialConditionSpec]:
    """
    Create initial conditions for model species.

    A starts at 1.0 in model units.
    Other species start at 0.0.

    The experimental signal scale is handled by ObservableSpec.
    """

    initial_condition_specs = []

    for species_name in model.species:
        if species_name == "A":
            initial_condition_specs.append(
                InitialConditionSpec(
                    species="A",
                    initial_guess=1.0,
                    lower_bound=0.0,
                    upper_bound=10.0,
                    fixed=True,
                    fixed_value=1.0,
                )
            )
        else:
            initial_condition_specs.append(
                InitialConditionSpec(
                    species=species_name,
                    initial_guess=0.0,
                    lower_bound=0.0,
                    upper_bound=10.0,
                    fixed=True,
                    fixed_value=0.0,
                )
            )

    return initial_condition_specs


def make_observable_specs() -> list[ObservableSpec]:
    """
    Map model species A to the observed amide percentage.

    amide_percent = scale * A + offset

    Since the first value is around 100, a reasonable initial guess is:
        scale ~ 20
        offset ~ 80

    This means the model can explain a partial decay from ~100 toward a
    nonzero plateau.
    """

    return [
        ObservableSpec(
            data_column="amide_percent",
            species="A",
            scale_initial_guess=20.0,
            scale_lower_bound=0.0,
            scale_upper_bound=200.0,
            scale_fixed=False,
            offset_initial_guess=80.0,
            offset_lower_bound=0.0,
            offset_upper_bound=120.0,
            offset_fixed=False,
        )
    ]


def fit_one_model(
    model_name: str,
    model_text: str,
    dataset: Dataset,
):
    """
    Fit one model to the amide percentage signal.
    """

    model = build_model_spec(model_text)

    parameter_specs = make_parameter_specs(model)
    initial_condition_specs = make_initial_condition_specs(model)
    observable_specs = make_observable_specs()

    settings = FitSettings(
        # Empty because observable_specs defines the mapping instead.
        species_mapping={},
        rtol=1e-8,
        atol=1e-10,
        max_nfev=2000,
    )

    result = fit_model(
        model=model,
        dataset=dataset,
        parameter_specs=parameter_specs,
        initial_condition_specs=initial_condition_specs,
        observable_specs=observable_specs,
        settings=settings,
    )

    model_output_dir = OUTPUT_DIR / model_name

    # For now, bundle plots still use direct species_mapping.
    # Because this fit uses observable_specs, skip plots here until
    # observable-aware plotting is implemented.
    export_fit_bundle(
        fit_result=result,
        model=model,
        dataset=dataset,
        output_dir=model_output_dir,
        parameter_specs=parameter_specs,
        initial_condition_specs=initial_condition_specs,
        observable_specs=observable_specs,
        species_mapping={"amide_percent": "A"},
        include_plots=True,
    )

    return result


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = read_real_amide_dataset(DATA_PATH)

    fit_results = {}

    for model_name, model_text in MODELS.items():
        print(f"\nFitting model: {model_name}")
        print(f"Model text: {model_text}")

        result = fit_one_model(
            model_name=model_name,
            model_text=model_text,
            dataset=dataset,
        )

        fit_results[model_name] = result

        print("Success:", result.success)
        print("Message:", result.message)
        print("Fitted kinetic parameters:", result.fitted_parameters)
        print("Fitted observable parameters:", result.fitted_observables)
        print("Statistics:", result.statistics)

    comparison_table = build_ranked_model_comparison_table(
        fit_results=fit_results,
        sort_by="aic",
    )

    comparison_path = OUTPUT_DIR / "model_comparison.csv"
    comparison_table.to_csv(comparison_path, index=False)

    print("\nModel comparison:")
    print(comparison_table)

    print(f"\nWrote outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
