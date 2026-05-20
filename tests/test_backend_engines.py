import numpy as np

from odefit.engines.registry import (
    available_engine_names,
    describe_available_engines,
    get_engine_bundle,
)
from odefit.model.model_spec import build_model_spec


def test_available_engine_names_contains_reference():
    names = available_engine_names()

    assert "reference" in names
    assert "numpy_scipy" in names


def test_get_engine_bundle_reference():
    bundle = get_engine_bundle("reference")

    assert bundle.name == "reference"
    assert bundle.solver is not None
    assert bundle.projection is not None
    assert bundle.least_squares is not None


def test_describe_available_engines():
    descriptions = describe_available_engines()

    names = {description["name"] for description in descriptions}

    assert "reference" in names


def test_reference_solver_engine_runs():
    bundle = get_engine_bundle("reference")

    model = build_model_spec("A -> B", name="single_step")

    result = bundle.solver.solve(
        model=model,
        parameters={"k1f": 0.4},
        initial_conditions={"A": 1.0, "B": 0.0},
        timepoints=np.array([0.0, 1.0, 2.0]),
    )

    assert result.success
    assert result.species == ["A", "B"]
    assert "A" in result.values
    assert "B" in result.values

    dataframe = result.to_dataframe()

    assert list(dataframe.columns) == ["time", "A", "B"]


def test_reference_single_species_projection_engine_runs():
    bundle = get_engine_bundle("reference")

    species_values = np.linspace(0.0, 1.0, 10)
    observed_values = 2.0 * species_values + 0.5

    result = bundle.projection.project_single_species(
        observed_values=observed_values,
        species_values=species_values,
        fit_scale=True,
        fit_offset=True,
    )

    assert abs(result.scale - 2.0) < 1e-10
    assert abs(result.offset - 0.5) < 1e-10
    assert result.rss < 1e-20


def test_reference_multispecies_projection_engine_runs():
    bundle = get_engine_bundle("reference")

    x1 = np.linspace(0.0, 1.0, 10)
    x2 = 1.0 - x1

    species_matrix = np.column_stack([x1, x2])
    observed_values = 2.0 * x1 - 0.5 * x2 + 0.25

    result = bundle.projection.project_multispecies(
        observed_values=observed_values,
        species_matrix=species_matrix,
        species_names=["A", "B"],
        fit_offset=True,
    )

    assert "A" in result.coefficients
    assert "B" in result.coefficients
    assert result.rss < 1e-20


def test_reference_least_squares_engine_runs():
    bundle = get_engine_bundle("reference")

    def residual_function(x):
        return np.array([x[0] - 2.0])

    result = bundle.least_squares.least_squares(
        residual_function,
        x0=np.array([0.0]),
        bounds=(
            np.array([-10.0]),
            np.array([10.0]),
        ),
    )

    assert result.success
    assert abs(result.x[0] - 2.0) < 1e-6
