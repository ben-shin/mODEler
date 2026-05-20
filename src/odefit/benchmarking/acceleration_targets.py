from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import inspect
import json
import statistics
import time
from typing import Callable, Any

import numpy as np
import pandas as pd

from odefit.api.backend import (
    bootstrap_global_observables_from_config,
    compare_global_observables_from_config,
    fit_global_observables_from_config,
    profile_likelihood_global_observables_from_config,
)
from odefit.data.dataset import Dataset
from odefit.fitting.fit_settings import FitSettings
from odefit.fitting.initial_condition_spec import InitialConditionSpec
from odefit.fitting.multispecies_variable_projection import (
    fit_global_observable_model_multispecies_variable_projection,
    solve_multispecies_observable_projection,
)
from odefit.fitting.parameter_spec import ParameterSpec
from odefit.fitting.variable_projection import (
    fit_global_observable_model_variable_projection,
    solve_scale_offset,
)
from odefit.model.model_spec import build_model_spec
from odefit.simulation.solver import simulate_model


@dataclass
class BenchmarkResult:
    name: str
    n_repeats: int
    times_seconds: list[float]
    metadata: dict[str, Any]

    @property
    def min_seconds(self) -> float:
        return min(self.times_seconds)

    @property
    def max_seconds(self) -> float:
        return max(self.times_seconds)

    @property
    def mean_seconds(self) -> float:
        return statistics.mean(self.times_seconds)

    @property
    def median_seconds(self) -> float:
        return statistics.median(self.times_seconds)

    @property
    def stdev_seconds(self) -> float:
        if len(self.times_seconds) < 2:
            return 0.0

        return statistics.stdev(self.times_seconds)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n_repeats": self.n_repeats,
            "times_seconds": self.times_seconds,
            "min_seconds": self.min_seconds,
            "max_seconds": self.max_seconds,
            "mean_seconds": self.mean_seconds,
            "median_seconds": self.median_seconds,
            "stdev_seconds": self.stdev_seconds,
            "metadata": self.metadata,
        }


def _time_callable(
    *,
    name: str,
    function: Callable[[], Any],
    n_repeats: int,
    metadata: dict[str, Any] | None = None,
) -> BenchmarkResult:
    times = []

    for _ in range(n_repeats):
        start = time.perf_counter()
        function()
        end = time.perf_counter()

        times.append(end - start)

    return BenchmarkResult(
        name=name,
        n_repeats=n_repeats,
        times_seconds=times,
        metadata=metadata or {},
    )

def _call_solve_scale_offset(
    *,
    species_values: np.ndarray,
    observed_values: np.ndarray,
    fit_scale: bool = True,
    fit_offset: bool = True,
):
    """
    Compatibility wrapper for solve_scale_offset.

    The exact solve_scale_offset signature has changed during development,
    so benchmark code should not rely on one keyword-name version.
    """

    signature = inspect.signature(solve_scale_offset)
    parameter_names = list(signature.parameters)

    if "species_values" in parameter_names:
        return solve_scale_offset(
            species_values=species_values,
            observed_values=observed_values,
            fit_scale=fit_scale,
            fit_offset=fit_offset,
        )

    if "observed_values" in parameter_names:
        return solve_scale_offset(
            observed_values=observed_values,
            species_values=species_values,
            fit_scale=fit_scale,
            fit_offset=fit_offset,
        )

    if "species_trace" in parameter_names:
        return solve_scale_offset(
            species_trace=species_values,
            observed_values=observed_values,
            fit_scale=fit_scale,
            fit_offset=fit_offset,
        )

    if "simulated_values" in parameter_names:
        return solve_scale_offset(
            simulated_values=species_values,
            observed_values=observed_values,
            fit_scale=fit_scale,
            fit_offset=fit_offset,
        )

    # Fallback for simple positional signatures.
    try:
        return solve_scale_offset(
            species_values,
            observed_values,
            fit_scale=fit_scale,
            fit_offset=fit_offset,
        )
    except TypeError:
        return solve_scale_offset(
            observed_values,
            species_values,
            fit_scale=fit_scale,
            fit_offset=fit_offset,
        )

def make_synthetic_global_observable_dataset(
    *,
    n_timepoints: int = 25,
    n_peaks: int = 50,
    k: float = 0.4,
    noise: float = 0.002,
    random_seed: int = 123,
) -> Dataset:
    timepoints = np.linspace(0.0, 10.0, n_timepoints)
    signal = np.exp(-k * timepoints)

    rng = np.random.default_rng(random_seed)

    dataframe = pd.DataFrame({"time": timepoints})

    for i in range(n_peaks):
        scale = rng.uniform(0.5, 2.0)
        offset = rng.uniform(-0.1, 0.1)

        dataframe[f"peak_{i}"] = (
            scale * signal
            + offset
            + rng.normal(0.0, noise, size=len(timepoints))
        )

    return Dataset(
        raw_dataframe=dataframe,
        time_column="time",
        signal_columns=[f"peak_{i}" for i in range(n_peaks)],
    )


def make_synthetic_multispecies_dataset(
    *,
    n_timepoints: int = 25,
    n_peaks: int = 50,
    k: float = 0.4,
    noise: float = 0.002,
    random_seed: int = 123,
) -> Dataset:
    timepoints = np.linspace(0.0, 10.0, n_timepoints)
    species_a = np.exp(-k * timepoints)
    species_b = 1.0 - species_a

    rng = np.random.default_rng(random_seed)

    dataframe = pd.DataFrame({"time": timepoints})

    for i in range(n_peaks):
        coefficient_a = rng.uniform(0.5, 2.0)
        coefficient_b = rng.uniform(-1.0, 1.0)
        offset = rng.uniform(-0.1, 0.1)

        dataframe[f"peak_{i}"] = (
            coefficient_a * species_a
            + coefficient_b * species_b
            + offset
            + rng.normal(0.0, noise, size=len(timepoints))
        )

    return Dataset(
        raw_dataframe=dataframe,
        time_column="time",
        signal_columns=[f"peak_{i}" for i in range(n_peaks)],
    )


def _single_step_model():
    return build_model_spec(
        "A -> B",
        name="single_step",
    )


def _parameter_specs() -> list[ParameterSpec]:
    return [
        ParameterSpec(
            name="k1f",
            initial_guess=0.2,
            lower_bound=1e-6,
            upper_bound=10.0,
        )
    ]


def _initial_condition_specs() -> list[InitialConditionSpec]:
    return [
        InitialConditionSpec(
            species="A",
            initial_guess=1.0,
            lower_bound=0.0,
            upper_bound=2.0,
            fixed=True,
            fixed_value=1.0,
        ),
        InitialConditionSpec(
            species="B",
            initial_guess=0.0,
            lower_bound=0.0,
            upper_bound=2.0,
            fixed=True,
            fixed_value=0.0,
        ),
    ]


def _fit_settings(max_nfev: int | None = 100) -> FitSettings:
    return FitSettings(
        species_mapping={},
        use_normalized_data=False,
        method="trf",
        loss="linear",
        max_nfev=max_nfev,
        rtol=1e-6,
        atol=1e-9,
    )


def benchmark_simulation_solve(
    *,
    n_timepoints: int = 100,
    n_repeats: int = 10,
) -> BenchmarkResult:
    model = _single_step_model()
    timepoints = np.linspace(0.0, 10.0, n_timepoints)

    def run():
        simulate_model(
            model=model,
            parameters={"k1f": 0.4},
            initial_conditions={"A": 1.0, "B": 0.0},
            timepoints=timepoints,
        )

    return _time_callable(
        name="simulation_solve",
        function=run,
        n_repeats=n_repeats,
        metadata={
            "n_timepoints": n_timepoints,
        },
    )


def benchmark_single_species_projection_kernel(
    *,
    n_timepoints: int = 100,
    n_repeats: int = 1000,
) -> BenchmarkResult:
    timepoints = np.linspace(0.0, 10.0, n_timepoints)
    species_values = np.exp(-0.4 * timepoints)
    observed = 1.7 * species_values + 0.2

    def run():
        _call_solve_scale_offset(
            species_values=species_values,
            observed_values=observed,
            fit_scale=True,
            fit_offset=True,
        )

    return _time_callable(
        name="single_species_projection_kernel",
        function=run,
        n_repeats=n_repeats,
        metadata={
            "n_timepoints": n_timepoints,
        },
    )


def benchmark_multispecies_projection_kernel(
    *,
    n_timepoints: int = 100,
    n_species: int = 3,
    n_repeats: int = 1000,
) -> BenchmarkResult:
    timepoints = np.linspace(0.0, 10.0, n_timepoints)

    columns = [
        np.exp(-(0.2 + 0.1 * i) * timepoints)
        for i in range(n_species)
    ]

    species_matrix = np.column_stack(columns)
    species_names = [f"S{i}" for i in range(n_species)]

    observed = species_matrix @ np.linspace(0.5, 1.5, n_species) + 0.1

    def run():
        solve_multispecies_observable_projection(
            signal=observed,
            species_matrix=species_matrix,
            species_names=species_names,
            fit_offset=True,
        )

    return _time_callable(
        name="multispecies_projection_kernel",
        function=run,
        n_repeats=n_repeats,
        metadata={
            "n_timepoints": n_timepoints,
            "n_species": n_species,
        },
    )


def benchmark_variable_projection_fit(
    *,
    n_timepoints: int = 25,
    n_peaks: int = 50,
    n_repeats: int = 3,
) -> BenchmarkResult:
    dataset = make_synthetic_global_observable_dataset(
        n_timepoints=n_timepoints,
        n_peaks=n_peaks,
    )

    def run():
        fit_global_observable_model_variable_projection(
            model=_single_step_model(),
            dataset=dataset,
            parameter_specs=_parameter_specs(),
            initial_condition_specs=_initial_condition_specs(),
            observed_species="A",
            settings=_fit_settings(max_nfev=100),
            signal_columns=dataset.signal_columns,
            fit_scale=True,
            fit_offset=True,
            backend="numpy",
            method="LSODA",
        )

    return _time_callable(
        name="variable_projection_fit",
        function=run,
        n_repeats=n_repeats,
        metadata={
            "n_timepoints": n_timepoints,
            "n_peaks": n_peaks,
        },
    )


def benchmark_multispecies_variable_projection_fit(
    *,
    n_timepoints: int = 25,
    n_peaks: int = 50,
    n_repeats: int = 3,
) -> BenchmarkResult:
    dataset = make_synthetic_multispecies_dataset(
        n_timepoints=n_timepoints,
        n_peaks=n_peaks,
    )

    def run():
        fit_global_observable_model_multispecies_variable_projection(
            model=_single_step_model(),
            dataset=dataset,
            parameter_specs=_parameter_specs(),
            initial_condition_specs=_initial_condition_specs(),
            observed_species=["A", "B"],
            settings=_fit_settings(max_nfev=100),
            signal_columns=dataset.signal_columns,
            fit_offset=True,
            backend="numpy",
            method="LSODA",
        )

    return _time_callable(
        name="multispecies_variable_projection_fit",
        function=run,
        n_repeats=n_repeats,
        metadata={
            "n_timepoints": n_timepoints,
            "n_peaks": n_peaks,
        },
    )


def _write_dataset_to_csv(dataset: Dataset, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.raw_dataframe.to_csv(path, index=False)

    return path


def make_api_fit_config(
    *,
    data_path: str,
    output_dir: str,
    multispecies: bool = False,
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "model_text": "A -> B",
        "data": data_path,
        "time_column": "time",
        "signal_columns": None,
        "exclude_columns": None,
        "output_dir": output_dir,
        "parameters": {
            "k1f": {
                "initial_guess": 0.2,
                "lower_bound": 1e-6,
                "upper_bound": 10.0,
            }
        },
        "initial_conditions": {
            "A": {
                "value": 1.0,
                "mode": "fixed",
            },
            "B": {
                "value": 0.0,
                "mode": "fixed",
            },
        },
        "method": "trf",
        "loss": "linear",
        "max_nfev": 100,
        "rtol": 1e-6,
        "atol": 1e-9,
        "variable_projection_backend": "numpy",
        "variable_projection_method": "LSODA",
        "show_progress": False,
        "no_plots": True,
    }

    if multispecies:
        config.update(
            {
                "use_multispecies_variable_projection": True,
                "observed_species": ["A", "B"],
                "fit_offset": True,
            }
        )
    else:
        config.update(
            {
                "use_variable_projection": True,
                "observed_species": "A",
                "fit_scale": True,
                "fit_offset": True,
            }
        )

    return config


def benchmark_api_bootstrap(
    *,
    work_dir: str | Path,
    n_timepoints: int = 25,
    n_peaks: int = 25,
    n_bootstrap: int = 5,
    n_workers: int = 1,
    n_repeats: int = 1,
    multispecies: bool = False,
) -> BenchmarkResult:
    work_path = Path(work_dir)

    dataset = (
        make_synthetic_multispecies_dataset(
            n_timepoints=n_timepoints,
            n_peaks=n_peaks,
        )
        if multispecies
        else make_synthetic_global_observable_dataset(
            n_timepoints=n_timepoints,
            n_peaks=n_peaks,
        )
    )

    data_path = _write_dataset_to_csv(
        dataset,
        work_path / "benchmark_bootstrap_data.csv",
    )

    config = make_api_fit_config(
        data_path=str(data_path),
        output_dir=str(work_path / "bootstrap_outputs"),
        multispecies=multispecies,
    )

    config.update(
        {
            "n_bootstrap": n_bootstrap,
            "n_workers": n_workers,
            "random_seed": 123,
            "confidence_level": 0.95,
        }
    )

    def run():
        bootstrap_global_observables_from_config(config)

    return _time_callable(
        name=(
            "multispecies_api_bootstrap"
            if multispecies
            else "single_species_api_bootstrap"
        ),
        function=run,
        n_repeats=n_repeats,
        metadata={
            "n_timepoints": n_timepoints,
            "n_peaks": n_peaks,
            "n_bootstrap": n_bootstrap,
            "n_workers": n_workers,
            "multispecies": multispecies,
        },
    )


def benchmark_api_profile_likelihood(
    *,
    work_dir: str | Path,
    n_timepoints: int = 25,
    n_peaks: int = 25,
    profile_n_points: int = 5,
    n_repeats: int = 1,
    multispecies: bool = False,
) -> BenchmarkResult:
    work_path = Path(work_dir)

    dataset = (
        make_synthetic_multispecies_dataset(
            n_timepoints=n_timepoints,
            n_peaks=n_peaks,
        )
        if multispecies
        else make_synthetic_global_observable_dataset(
            n_timepoints=n_timepoints,
            n_peaks=n_peaks,
        )
    )

    data_path = _write_dataset_to_csv(
        dataset,
        work_path / "benchmark_profile_data.csv",
    )

    config = make_api_fit_config(
        data_path=str(data_path),
        output_dir=str(work_path / "profile_outputs"),
        multispecies=multispecies,
    )

    config.update(
        {
            "profile_parameters": ["k1f"],
            "profile_n_points": profile_n_points,
            "profile_span_factor": 5.0,
            "profile_log_space": True,
        }
    )

    def run():
        profile_likelihood_global_observables_from_config(config)

    return _time_callable(
        name=(
            "multispecies_api_profile_likelihood"
            if multispecies
            else "single_species_api_profile_likelihood"
        ),
        function=run,
        n_repeats=n_repeats,
        metadata={
            "n_timepoints": n_timepoints,
            "n_peaks": n_peaks,
            "profile_n_points": profile_n_points,
            "multispecies": multispecies,
        },
    )


def benchmark_api_model_comparison(
    *,
    work_dir: str | Path,
    n_timepoints: int = 25,
    n_peaks: int = 25,
    n_repeats: int = 1,
    multispecies: bool = False,
) -> BenchmarkResult:
    work_path = Path(work_dir)

    dataset = (
        make_synthetic_multispecies_dataset(
            n_timepoints=n_timepoints,
            n_peaks=n_peaks,
        )
        if multispecies
        else make_synthetic_global_observable_dataset(
            n_timepoints=n_timepoints,
            n_peaks=n_peaks,
        )
    )

    data_path = _write_dataset_to_csv(
        dataset,
        work_path / "benchmark_model_comparison_data.csv",
    )

    config: dict[str, Any] = {
        "model_texts": {
            "single_step": "A -> B",
            "two_step": "A -> B\nB -> C",
        },
        "data": str(data_path),
        "time_column": "time",
        "signal_columns": None,
        "exclude_columns": None,
        "output_dir": str(work_path / "model_comparison_outputs"),
        "parameters_by_model": {
            "single_step": {
                "k1f": {
                    "initial_guess": 0.2,
                    "lower_bound": 1e-6,
                    "upper_bound": 10.0,
                }
            },
            "two_step": {
                "k1f": {
                    "initial_guess": 0.2,
                    "lower_bound": 1e-6,
                    "upper_bound": 10.0,
                },
                "k2f": {
                    "initial_guess": 0.1,
                    "lower_bound": 1e-6,
                    "upper_bound": 10.0,
                },
            },
        },
        "initial_conditions_by_model": {
            "single_step": {
                "A": {
                    "value": 1.0,
                    "mode": "fixed",
                },
                "B": {
                    "value": 0.0,
                    "mode": "fixed",
                },
            },
            "two_step": {
                "A": {
                    "value": 1.0,
                    "mode": "fixed",
                },
                "B": {
                    "value": 0.0,
                    "mode": "fixed",
                },
                "C": {
                    "value": 0.0,
                    "mode": "fixed",
                },
            },
        },
        "method": "trf",
        "loss": "linear",
        "max_nfev": 100,
        "rtol": 1e-6,
        "atol": 1e-9,
        "variable_projection_backend": "numpy",
        "variable_projection_method": "LSODA",
        "sort_by": "bic",
        "show_progress": False,
    }

    if multispecies:
        config.update(
            {
                "use_multispecies_variable_projection": True,
                "observed_species": ["A", "B"],
                "observed_species_by_model": {
                    "single_step": ["A", "B"],
                    "two_step": ["A", "B"],
                },
                "fit_offset": True,
            }
        )
    else:
        config.update(
            {
                "use_variable_projection": True,
                "observed_species": "A",
                "fit_scale": True,
                "fit_offset": True,
            }
        )

    def run():
        compare_global_observables_from_config(config)

    return _time_callable(
        name=(
            "multispecies_api_model_comparison"
            if multispecies
            else "single_species_api_model_comparison"
        ),
        function=run,
        n_repeats=n_repeats,
        metadata={
            "n_timepoints": n_timepoints,
            "n_peaks": n_peaks,
            "multispecies": multispecies,
        },
    )


def run_acceleration_target_benchmarks(
    *,
    work_dir: str | Path,
    n_repeats_fast: int = 5,
    n_repeats_fit: int = 2,
    include_slow: bool = False,
) -> list[BenchmarkResult]:
    work_path = Path(work_dir)
    work_path.mkdir(parents=True, exist_ok=True)

    results = [
        benchmark_simulation_solve(
            n_timepoints=100,
            n_repeats=n_repeats_fast,
        ),
        benchmark_single_species_projection_kernel(
            n_timepoints=100,
            n_repeats=max(10, n_repeats_fast),
        ),
        benchmark_multispecies_projection_kernel(
            n_timepoints=100,
            n_species=3,
            n_repeats=max(10, n_repeats_fast),
        ),
        benchmark_variable_projection_fit(
            n_timepoints=25,
            n_peaks=50,
            n_repeats=n_repeats_fit,
        ),
        benchmark_multispecies_variable_projection_fit(
            n_timepoints=25,
            n_peaks=50,
            n_repeats=n_repeats_fit,
        ),
        benchmark_api_model_comparison(
            work_dir=work_path,
            n_timepoints=25,
            n_peaks=25,
            n_repeats=1,
            multispecies=False,
        ),
        benchmark_api_model_comparison(
            work_dir=work_path,
            n_timepoints=25,
            n_peaks=25,
            n_repeats=1,
            multispecies=True,
        ),
    ]

    if include_slow:
        results.extend(
            [
                benchmark_api_bootstrap(
                    work_dir=work_path,
                    n_timepoints=25,
                    n_peaks=25,
                    n_bootstrap=5,
                    n_workers=1,
                    n_repeats=1,
                    multispecies=False,
                ),
                benchmark_api_bootstrap(
                    work_dir=work_path,
                    n_timepoints=25,
                    n_peaks=25,
                    n_bootstrap=5,
                    n_workers=1,
                    n_repeats=1,
                    multispecies=True,
                ),
                benchmark_api_profile_likelihood(
                    work_dir=work_path,
                    n_timepoints=25,
                    n_peaks=25,
                    profile_n_points=5,
                    n_repeats=1,
                    multispecies=False,
                ),
                benchmark_api_profile_likelihood(
                    work_dir=work_path,
                    n_timepoints=25,
                    n_peaks=25,
                    profile_n_points=5,
                    n_repeats=1,
                    multispecies=True,
                ),
            ]
        )

    return results


def export_benchmark_results(
    results: list[BenchmarkResult],
    output_dir: str | Path,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    json_path = output_path / "acceleration_target_benchmarks.json"
    csv_path = output_path / "acceleration_target_benchmarks.csv"

    payload = [result.to_dict() for result in results]

    with json_path.open("w") as handle:
        json.dump(
            payload,
            handle,
            indent=2,
        )

    rows = []

    for result in results:
        row = {
            "name": result.name,
            "n_repeats": result.n_repeats,
            "min_seconds": result.min_seconds,
            "max_seconds": result.max_seconds,
            "mean_seconds": result.mean_seconds,
            "median_seconds": result.median_seconds,
            "stdev_seconds": result.stdev_seconds,
        }

        for key, value in result.metadata.items():
            row[f"metadata_{key}"] = value

        rows.append(row)

    pd.DataFrame(rows).to_csv(
        csv_path,
        index=False,
    )

    return {
        "json": json_path,
        "csv": csv_path,
    }
