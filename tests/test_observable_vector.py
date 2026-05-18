import numpy as np
import pandas as pd
import pytest

from odefit.data.dataset import Dataset
from odefit.fitting.observable_spec import ObservableSpec
from odefit.fitting.observable_vector import (
    build_initial_observable_parameters,
    build_observable_bounds,
    build_observable_vector,
    get_free_observable_parameter_names,
    make_observable_specs_from_species_mapping,
    validate_observable_specs,
    vector_to_observable_parameters,
)
from odefit.model.model_spec import build_model_spec


def make_dataset():
    return Dataset(
        raw_dataframe=pd.DataFrame(
            {
                "time": [0.0, 1.0],
                "amide": [1.0, 0.5],
            }
        ),
        time_column="time",
        signal_columns=["amide"],
    )


def test_make_observable_specs_from_species_mapping():
    specs = make_observable_specs_from_species_mapping({"amide": "A"})

    assert len(specs) == 1
    assert specs[0].data_column == "amide"
    assert specs[0].species == "A"
    assert specs[0].scale_fixed is True
    assert specs[0].scale_fixed_value == 1.0
    assert specs[0].offset_fixed is True
    assert specs[0].offset_fixed_value == 0.0


def test_get_free_observable_parameter_names():
    specs = [
        ObservableSpec(
            data_column="amide",
            species="A",
            scale_fixed=False,
            offset_fixed=False,
        )
    ]

    names = get_free_observable_parameter_names(specs)

    assert names == ["amide_scale", "amide_offset"]


def test_build_observable_vector():
    specs = [
        ObservableSpec(
            data_column="amide",
            species="A",
            scale_initial_guess=2.0,
            scale_fixed=False,
            offset_initial_guess=0.1,
            offset_fixed=False,
        )
    ]

    vector = build_observable_vector(specs)

    assert np.allclose(vector, [2.0, 0.1])


def test_build_observable_bounds():
    specs = [
        ObservableSpec(
            data_column="amide",
            species="A",
            scale_lower_bound=0.0,
            scale_upper_bound=10.0,
            scale_fixed=False,
            offset_lower_bound=-1.0,
            offset_upper_bound=1.0,
            offset_fixed=False,
        )
    ]

    lower, upper = build_observable_bounds(specs)

    assert np.allclose(lower, [0.0, -1.0])
    assert np.allclose(upper, [10.0, 1.0])


def test_vector_to_observable_parameters():
    specs = [
        ObservableSpec(
            data_column="amide",
            species="A",
            scale_fixed=False,
            offset_fixed=False,
        )
    ]

    parameters = vector_to_observable_parameters(
        vector=np.array([2.0, 0.1]),
        observable_specs=specs,
    )

    assert parameters == {
        "amide": {
            "species": "A",
            "scale": 2.0,
            "offset": 0.1,
        }
    }


def test_vector_to_observable_parameters_uses_fixed_values():
    specs = [
        ObservableSpec(
            data_column="amide",
            species="A",
            scale_fixed=True,
            scale_fixed_value=3.0,
            offset_fixed=True,
            offset_fixed_value=0.2,
        )
    ]

    parameters = vector_to_observable_parameters(
        vector=np.array([]),
        observable_specs=specs,
    )

    assert parameters == {
        "amide": {
            "species": "A",
            "scale": 3.0,
            "offset": 0.2,
        }
    }


def test_build_initial_observable_parameters():
    specs = [
        ObservableSpec(
            data_column="amide",
            species="A",
            scale_initial_guess=2.0,
            scale_fixed=False,
            offset_initial_guess=0.1,
            offset_fixed=False,
        )
    ]

    parameters = build_initial_observable_parameters(specs)

    assert parameters == {
        "amide": {
            "species": "A",
            "scale": 2.0,
            "offset": 0.1,
        }
    }


def test_short_observable_vector_raises_error():
    specs = [
        ObservableSpec(
            data_column="amide",
            species="A",
            scale_fixed=False,
            offset_fixed=False,
        )
    ]

    with pytest.raises(ValueError):
        vector_to_observable_parameters(
            vector=np.array([2.0]),
            observable_specs=specs,
        )


def test_long_observable_vector_raises_error():
    specs = [
        ObservableSpec(
            data_column="amide",
            species="A",
            scale_fixed=False,
            offset_fixed=True,
            offset_fixed_value=0.0,
        )
    ]

    with pytest.raises(ValueError):
        vector_to_observable_parameters(
            vector=np.array([2.0, 0.1]),
            observable_specs=specs,
        )


def test_validate_observable_specs_accepts_valid_spec():
    model = build_model_spec("A>B")
    dataset = make_dataset()

    validate_observable_specs(
        model=model,
        dataset=dataset,
        observable_specs=[
            ObservableSpec(
                data_column="amide",
                species="A",
            )
        ],
    )


def test_validate_observable_specs_rejects_missing_data_column():
    model = build_model_spec("A>B")
    dataset = make_dataset()

    with pytest.raises(ValueError):
        validate_observable_specs(
            model=model,
            dataset=dataset,
            observable_specs=[
                ObservableSpec(
                    data_column="missing",
                    species="A",
                )
            ],
        )


def test_validate_observable_specs_rejects_missing_species():
    model = build_model_spec("A>B")
    dataset = make_dataset()

    with pytest.raises(ValueError):
        validate_observable_specs(
            model=model,
            dataset=dataset,
            observable_specs=[
                ObservableSpec(
                    data_column="amide",
                    species="C",
                )
            ],
        )


def test_validate_observable_specs_rejects_duplicate_data_column():
    model = build_model_spec("A>B")
    dataset = make_dataset()

    with pytest.raises(ValueError):
        validate_observable_specs(
            model=model,
            dataset=dataset,
            observable_specs=[
                ObservableSpec("amide", "A"),
                ObservableSpec("amide", "B"),
            ],
        )
