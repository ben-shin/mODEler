from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Any

import numpy as np
import pandas as pd

from odefit.data.dataset import Dataset
from odefit.fitting.fit_settings import FitSettings
from odefit.fitting.global_observable_multistart import (
    fit_global_observable_multistart,
)
from odefit.fitting.global_observables import (
    build_shared_species_observable_specs,
    fit_global_observable_model,
)
from odefit.fitting.initial_condition_spec import InitialConditionSpec
from odefit.fitting.optimizer import fit_model
from odefit.fitting.parameter_spec import ParameterSpec
from odefit.model.model_spec import build_model_spec


@dataclass
class BenchmarkResult:
    """
    Result from one benchmark run.
    """

    name: str
    elapsed_seconds: float
    metadata: dict[str, Any]


def benchmark_callable(
    name: str,
    function: Callable[[], Any],
    metadata: dict[str, Any] | None = None,
) -> BenchmarkResult:
    """
    Time a callable and return elapsed runtime.
    """

    start = time.perf_counter()
    function()
    end = time.perf_counter()

    return BenchmarkResult(
        name=name,
        elapsed_seconds=end - start,
        metadata={} if metadata is None else metadata,
    )


def benchmark_results_to_dataframe(
    results: list[BenchmarkResult],
) -> pd.DataFrame:
    """
    Convert benchmark results to a dataframe.
    """

    rows: list[dict[str, Any]] = []

    for result in results:
        row = {
            "name": result.name,
            "elapsed_seconds": result.elapsed_seconds,
        }

        row.update(result.metadata)
        rows.append(row)

    return pd.DataFrame(rows)


def write_benchmark_results(
    results: list[BenchmarkResult],
    output_path: str | Path,
) -> Path:
    """
    Write benchmark results to CSV.
    """

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    table = benchmark_results_to_dataframe(results)
    table.to_csv(path, index=False)

    return path


def make_first_order_dataset(
    n_timepoints: int = 30,
    true_k: float = 0.4,
) -> Dataset:
    """
    Build a simple A -> B dataset for standard fitting benchmarks.
    """

    timepoints = np.linspace(0.0, 8.0, n_timepoints)
    a_values = np.exp(-true_k * timepoints)
    b_values = 1.0 - a_values

    dataframe = pd.DataFrame(
        {
            "time": timepoints,
            "A": a_values,
            "B": b_values,
        }
    )

    return Dataset(
        raw_dataframe=dataframe,
        time_column="time",
        signal_columns=["A", "B"],
    )


def make_hsqc_like_dataset(
    n_peaks: int = 25,
    n_timepoints: int = 30,
    true_k: float = 0.4,
    random_seed: int = 1,
) -> Dataset:
    """
    Build synthetic wide-format HSQC-like peak intensity data.

    Each peak observes the same species A with its own scale and offset:

        peak_i(t) = scale_i * A(t) + offset_i
    """

    rng = np.random.default_rng(random_seed)

    timepoints = np.linspace(0.0, 8.0, n_timepoints)
    a_values = np.exp(-true_k * timepoints)

    dataframe_data: dict[str, Any] = {
        "time": timepoints,
    }

    for peak_index in range(n_peaks):
        scale = rng.uniform(0.5, 3.0)
        offset = rng.uniform(0.0, 0.25)

        column_name = f"P{peak_index + 1}"

        dataframe_data[column_name] = scale * a_values + offset

    dataframe = pd.DataFrame(dataframe_data)

    signal_columns = [
        column
        for column in dataframe.columns
        if column != "time"
    ]

    return Dataset(
        raw_dataframe=dataframe,
        time_column="time",
        signal_columns=signal_columns,
    )


def make_first_order_parameter_specs() -> list[ParameterSpec]:
    """
    Parameter specs for A -> B.
    """

    return [
        ParameterSpec(
            name="k1f",
            initial_guess=0.1,
            lower_bound=0.001,
            upper_bound=10.0,
        )
    ]


def make_first_order_initial_condition_specs() -> list[InitialConditionSpec]:
    """
    Initial condition specs for A -> B.
    """

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


def benchmark_standard_fit(
    n_timepoints: int = 30,
) -> BenchmarkResult:
    """
    Benchmark ordinary species-to-column fitting.
    """

    model = build_model_spec("A>B")
    dataset = make_first_order_dataset(n_timepoints=n_timepoints)

    parameter_specs = make_first_order_parameter_specs()
    initial_condition_specs = make_first_order_initial_condition_specs()

    settings = FitSettings(
        species_mapping={
            "A": "A",
            "B": "B",
        },
        rtol=1e-8,
        atol=1e-10,
    )

    def run() -> None:
        fit_model(
            model=model,
            dataset=dataset,
            parameter_specs=parameter_specs,
            initial_condition_specs=initial_condition_specs,
            settings=settings,
        )

    return benchmark_callable(
        name="standard_fit",
        function=run,
        metadata={
            "n_timepoints": n_timepoints,
            "n_peaks": None,
            "n_starts": None,
            "n_workers": None,
        },
    )


def benchmark_global_observable_fit(
    n_peaks: int = 25,
    n_timepoints: int = 30,
) -> BenchmarkResult:
    """
    Benchmark global observable fitting.
    """

    model = build_model_spec("A>B")
    dataset = make_hsqc_like_dataset(
        n_peaks=n_peaks,
        n_timepoints=n_timepoints,
    )

    parameter_specs = make_first_order_parameter_specs()
    initial_condition_specs = make_first_order_initial_condition_specs()

    settings = FitSettings(
        species_mapping={},
        rtol=1e-8,
        atol=1e-10,
    )

    def run() -> None:
        fit_global_observable_model(
            model=model,
            dataset=dataset,
            parameter_specs=parameter_specs,
            initial_condition_specs=initial_condition_specs,
            observed_species="A",
            settings=settings,
            fit_scale=True,
            fit_offset=True,
            scale_initial_guess=1.0,
            scale_lower_bound=0.0,
            scale_upper_bound=5.0,
            offset_initial_guess=0.0,
            offset_lower_bound=-1.0,
            offset_upper_bound=1.0,
        )

    return benchmark_callable(
        name="global_observable_fit",
        function=run,
        metadata={
            "n_timepoints": n_timepoints,
            "n_peaks": n_peaks,
            "n_starts": None,
            "n_workers": None,
        },
    )


def benchmark_global_observable_multistart(
    n_peaks: int = 25,
    n_timepoints: int = 30,
    n_starts: int = 4,
    n_workers: int = 1,
) -> BenchmarkResult:
    """
    Benchmark global observable multistart fitting.
    """

    model = build_model_spec("A>B")
    dataset = make_hsqc_like_dataset(
        n_peaks=n_peaks,
        n_timepoints=n_timepoints,
    )

    parameter_specs = make_first_order_parameter_specs()
    initial_condition_specs = make_first_order_initial_condition_specs()

    observable_specs = build_shared_species_observable_specs(
        signal_columns=dataset.signal_columns,
        species="A",
        fit_scale=True,
        fit_offset=True,
        scale_initial_guess=1.0,
        scale_lower_bound=0.0,
        scale_upper_bound=5.0,
        offset_initial_guess=0.0,
        offset_lower_bound=-1.0,
        offset_upper_bound=1.0,
    )

    settings = FitSettings(
        species_mapping={},
        rtol=1e-8,
        atol=1e-10,
    )

    def run() -> None:
        fit_global_observable_multistart(
            model=model,
            dataset=dataset,
            parameter_specs=parameter_specs,
            initial_condition_specs=initial_condition_specs,
            observable_specs=observable_specs,
            settings=settings,
            n_starts=n_starts,
            n_workers=n_workers,
            random_seed=1,
            sort_by="aic",
            show_progress=False,
        )

    return benchmark_callable(
        name="global_observable_multistart",
        function=run,
        metadata={
            "n_timepoints": n_timepoints,
            "n_peaks": n_peaks,
            "n_starts": n_starts,
            "n_workers": n_workers,
        },
    )


def run_default_benchmarks() -> list[BenchmarkResult]:
    """
    Run a small default benchmark suite.

    This is intentionally modest so it can run during development.
    """

    results = [
        benchmark_standard_fit(n_timepoints=30),
        benchmark_global_observable_fit(n_peaks=10, n_timepoints=30),
        benchmark_global_observable_fit(n_peaks=50, n_timepoints=30),
        benchmark_global_observable_multistart(
            n_peaks=10,
            n_timepoints=30,
            n_starts=4,
            n_workers=1,
        ),
    ]

    return results
