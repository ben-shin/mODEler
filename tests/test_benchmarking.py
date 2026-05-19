import pandas as pd

from odefit.performance.benchmarking import (
    BenchmarkResult,
    benchmark_callable,
    benchmark_results_to_dataframe,
    make_first_order_dataset,
    make_hsqc_like_dataset,
    write_benchmark_results,
)


def test_benchmark_callable():
    result = benchmark_callable(
        name="test",
        function=lambda: sum([1, 2, 3]),
        metadata={
            "kind": "unit",
        },
    )

    assert result.name == "test"
    assert result.elapsed_seconds >= 0.0
    assert result.metadata == {"kind": "unit"}


def test_benchmark_results_to_dataframe():
    results = [
        BenchmarkResult(
            name="one",
            elapsed_seconds=0.1,
            metadata={
                "n_peaks": 10,
            },
        ),
        BenchmarkResult(
            name="two",
            elapsed_seconds=0.2,
            metadata={
                "n_peaks": 20,
            },
        ),
    ]

    table = benchmark_results_to_dataframe(results)

    assert list(table.columns) == [
        "name",
        "elapsed_seconds",
        "n_peaks",
    ]

    assert list(table["name"]) == ["one", "two"]
    assert list(table["n_peaks"]) == [10, 20]


def test_write_benchmark_results(tmp_path):
    output_path = tmp_path / "benchmarks.csv"

    results = [
        BenchmarkResult(
            name="one",
            elapsed_seconds=0.1,
            metadata={
                "n_peaks": 10,
            },
        )
    ]

    written_path = write_benchmark_results(
        results=results,
        output_path=output_path,
    )

    assert written_path == output_path
    assert output_path.exists()

    table = pd.read_csv(output_path)

    assert table["name"].iloc[0] == "one"
    assert table["n_peaks"].iloc[0] == 10


def test_make_first_order_dataset():
    dataset = make_first_order_dataset(n_timepoints=5)

    assert dataset.time_column == "time"
    assert dataset.signal_columns == ["A", "B"]
    assert len(dataset.raw_dataframe) == 5


def test_make_hsqc_like_dataset():
    dataset = make_hsqc_like_dataset(
        n_peaks=4,
        n_timepoints=5,
    )

    assert dataset.time_column == "time"
    assert dataset.signal_columns == ["P1", "P2", "P3", "P4"]
    assert len(dataset.raw_dataframe) == 5
