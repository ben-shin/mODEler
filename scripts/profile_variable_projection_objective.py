from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from odefit.data.dataset import Dataset
from odefit.engines.registry import get_engine_bundle
from odefit.fitting.engine_helpers import (
    engine_project_single_species_batch,
    engine_solve_to_dataframe,
)
from odefit.fitting.fit_settings import FitSettings
from odefit.fitting.initial_condition_spec import InitialConditionSpec
from odefit.fitting.parameter_spec import ParameterSpec
from odefit.fitting.variable_projection import project_observables_onto_species
from odefit.model.model_spec import build_model_spec


@dataclass
class ObjectiveProfileRow:
    engine_name: str
    n_timepoints: int
    n_observables: int
    n_repeats: int
    solve_median_seconds: float
    solve_mean_seconds: float
    projection_batch_median_seconds: float
    projection_batch_mean_seconds: float
    full_projection_assembly_median_seconds: float
    full_projection_assembly_mean_seconds: float
    residual_vector_length: int
    rss: float


def make_dataset(
    *,
    n_timepoints: int,
    n_observables: int,
    random_seed: int,
) -> Dataset:
    rng = np.random.default_rng(random_seed)

    timepoints = np.linspace(0.0, 10.0, n_timepoints)
    species_a = np.exp(-0.4 * timepoints)

    scales = rng.uniform(0.5, 2.0, size=n_observables)
    offsets = rng.uniform(-0.1, 0.1, size=n_observables)

    data = {"time": timepoints}
    signal_columns = []

    for index in range(n_observables):
        column = f"peak_{index}"
        signal_columns.append(column)
        data[column] = (
            scales[index] * species_a
            + offsets[index]
            + rng.normal(0.0, 0.002, size=n_timepoints)
        )

    dataframe = pd.DataFrame(data)

    return Dataset(
        raw_dataframe=dataframe,
        time_column="time",
        signal_columns=signal_columns,
    )


def _time_call(function, *, n_repeats: int):
    values = []
    last_result = None

    for _ in range(n_repeats):
        start = time.perf_counter()
        last_result = function()
        end = time.perf_counter()

        values.append(end - start)

    return values, last_result


def profile_objective_components(
    *,
    engine_name: str,
    n_timepoints: int,
    n_observables: int,
    n_repeats: int,
    random_seed: int,
) -> ObjectiveProfileRow:
    dataset = make_dataset(
        n_timepoints=n_timepoints,
        n_observables=n_observables,
        random_seed=random_seed,
    )

    model = build_model_spec(
        "A -> B",
        name="single_step",
    )

    settings = FitSettings(
        species_mapping={},
        use_normalized_data=False,
        method="trf",
        loss="linear",
        max_nfev=20,
        rtol=1e-6,
        atol=1e-9,
    )

    parameter_specs = [
        ParameterSpec(
            name="k1f",
            initial_guess=0.4,
            lower_bound=1e-6,
            upper_bound=10.0,
        )
    ]

    initial_condition_specs = [
        InitialConditionSpec(
            species="A",
            initial_guess=1.0,
            fixed=True,
            fixed_value=1.0,
        ),
        InitialConditionSpec(
            species="B",
            initial_guess=0.0,
            fixed=True,
            fixed_value=0.0,
        ),
    ]

    parameters = {
        spec.name: spec.initial_guess
        for spec in parameter_specs
    }

    initial_conditions = {
        spec.species: spec.fixed_value if spec.fixed else spec.initial_guess
        for spec in initial_condition_specs
    }

    timepoints = dataset.time_values
    engine_bundle = get_engine_bundle(engine_name)

    solve_times, simulation_dataframe = _time_call(
        lambda: engine_solve_to_dataframe(
            engine_bundle=engine_bundle,
            model=model,
            parameters=parameters,
            initial_conditions=initial_conditions,
            timepoints=timepoints,
            settings=settings.to_simulation_settings()
            if hasattr(settings, "to_simulation_settings")
            else None,
        ),
        n_repeats=n_repeats,
    )

    species_values = simulation_dataframe["A"].to_numpy(dtype=float)

    observed_matrix = dataset.raw_dataframe[dataset.signal_columns].to_numpy(
        dtype=float,
    )

    projection_batch_times, projection_result = _time_call(
        lambda: engine_project_single_species_batch(
            engine_bundle=engine_bundle,
            observed_matrix=observed_matrix,
            species_values=species_values,
            fit_scale=True,
            fit_offset=True,
        ),
        n_repeats=n_repeats,
    )

    full_projection_times, full_projection_result = _time_call(
        lambda: project_observables_onto_species(
            timepoints=timepoints,
            simulated_species_values=species_values,
            observed_dataframe=dataset.raw_dataframe,
            signal_columns=dataset.signal_columns,
            fit_scale=True,
            fit_offset=True,
            engine_bundle=engine_bundle,
        ),
        n_repeats=n_repeats,
    )

    return ObjectiveProfileRow(
        engine_name=engine_name,
        n_timepoints=n_timepoints,
        n_observables=n_observables,
        n_repeats=n_repeats,
        solve_median_seconds=float(np.median(solve_times)),
        solve_mean_seconds=float(np.mean(solve_times)),
        projection_batch_median_seconds=float(np.median(projection_batch_times)),
        projection_batch_mean_seconds=float(np.mean(projection_batch_times)),
        full_projection_assembly_median_seconds=float(np.median(full_projection_times)),
        full_projection_assembly_mean_seconds=float(np.mean(full_projection_times)),
        residual_vector_length=int(len(full_projection_result.residual_vector)),
        rss=float(full_projection_result.rss),
    )


def add_fraction_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    output = dataframe.copy()

    output["projection_plus_assembly_over_solve_median"] = (
        output["full_projection_assembly_median_seconds"]
        / output["solve_median_seconds"]
    )

    output["assembly_over_batch_projection_median"] = (
        output["full_projection_assembly_median_seconds"]
        / output["projection_batch_median_seconds"]
    )

    output["batch_projection_fraction_of_full_projection_assembly"] = (
        output["projection_batch_median_seconds"]
        / output["full_projection_assembly_median_seconds"]
    )

    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile variable-projection objective components."
    )

    parser.add_argument(
        "--engines",
        nargs="+",
        default=["reference", "numba_projection", "jax_projection"],
    )

    parser.add_argument("--n-timepoints", type=int, default=50)
    parser.add_argument("--n-observables", type=int, default=100)
    parser.add_argument("--n-repeats", type=int, default=50)
    parser.add_argument("--random-seed", type=int, default=123)
    parser.add_argument(
        "--output-dir",
        default="benchmarks/variable_projection_objective_profile",
    )

    args = parser.parse_args()

    rows = []

    for engine_name in args.engines:
        try:
            row = profile_objective_components(
                engine_name=engine_name,
                n_timepoints=args.n_timepoints,
                n_observables=args.n_observables,
                n_repeats=args.n_repeats,
                random_seed=args.random_seed,
            )

            print(
                f"{engine_name}: "
                f"solve={row.solve_median_seconds:.6f}s "
                f"batch_projection={row.projection_batch_median_seconds:.6f}s "
                f"full_projection_assembly={row.full_projection_assembly_median_seconds:.6f}s "
                f"rss={row.rss:.6g}"
            )

            rows.append(asdict(row))

        except Exception as exc:
            print(f"{engine_name}: unavailable ({type(exc).__name__}: {exc})")
            rows.append(
                {
                    "engine_name": engine_name,
                    "available": False,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )

    dataframe = pd.DataFrame(rows)

    if not dataframe.empty and "solve_median_seconds" in dataframe.columns:
        dataframe = add_fraction_columns(dataframe)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "variable_projection_objective_profile.json"
    csv_path = output_dir / "variable_projection_objective_profile.csv"

    with json_path.open("w") as handle:
        json.dump(
            dataframe.where(
                pd.notna(dataframe),
                None,
            ).to_dict(orient="records"),
            handle,
            indent=2,
        )

    dataframe.to_csv(csv_path, index=False)

    print("\nWrote JSON:", json_path)
    print("Wrote CSV:", csv_path)


if __name__ == "__main__":
    main()
