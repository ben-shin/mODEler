import numpy as np
import pandas as pd
import pytest

from odefit.data.dataset import Dataset
from odefit.fitting.fit_settings import FitSettings
from odefit.fitting.global_observables import (
    build_shared_species_observable_specs,
    build_signal_weights_from_columns,
    fit_global_observable_model,
    infer_signal_columns_from_dataframe,
    read_wide_observable_dataset,
)
from odefit.fitting.initial_condition_spec import InitialConditionSpec
from odefit.fitting.parameter_spec import ParameterSpec
from odefit.model.model_spec import build_model_spec


def test_infer_signal_columns_from_dataframe():
    dataframe = pd.DataFrame(
        {
            "time": [0.0, 1.0],
            "assignment": ["A23", "A23"],
            "A23_HN": [1000.0, 900.0],
            "G45_HN": [800.0, 760.0],
        }
    )

    signal_columns = infer_signal_columns_from_dataframe(
        dataframe=dataframe,
        time_column="time",
        exclude_columns=["assignment"],
    )

    assert signal_columns == ["A23_HN", "G45_HN"]


def test_infer_signal_columns_rejects_missing_time_column():
    dataframe = pd.DataFrame(
        {
            "A23_HN": [1000.0, 900.0],
        }
    )

    with pytest.raises(ValueError):
        infer_signal_columns_from_dataframe(
            dataframe=dataframe,
            time_column="time",
        )


def test_read_wide_observable_dataset(tmp_path):
    data_path = tmp_path / "hsqc.csv"

    dataframe = pd.DataFrame(
        {
            "time": [0.0, 1.0, 2.0],
            "A23_HN": [1000.0, 900.0, 850.0],
            "G45_HN": [800.0, 760.0, 720.0],
        }
    )

    dataframe.to_csv(data_path, index=False)

    dataset = read_wide_observable_dataset(
        file_path=data_path,
        time_column="time",
    )

    assert dataset.time_column == "time"
    assert dataset.signal_columns == ["A23_HN", "G45_HN"]
    assert np.allclose(dataset.time_values, [0.0, 1.0, 2.0])


def test_build_shared_species_observable_specs():
    specs = build_shared_species_observable_specs(
        signal_columns=["A23_HN", "G45_HN"],
        species="A",
        scale_initial_guess=100.0,
        scale_lower_bound=0.0,
        scale_upper_bound=2000.0,
        offset_initial_guess=0.0,
        offset_lower_bound=-100.0,
        offset_upper_bound=100.0,
    )

    assert len(specs) == 2

    assert specs[0].data_column == "A23_HN"
    assert specs[0].species == "A"
    assert specs[0].scale_fixed is False
    assert specs[0].offset_fixed is False

    assert specs[1].data_column == "G45_HN"
    assert specs[1].species == "A"


def test_build_shared_species_observable_specs_can_fix_scale_and_offset():
    specs = build_shared_species_observable_specs(
        signal_columns=["A23_HN"],
        species="A",
        fit_scale=False,
        fit_offset=False,
        scale_initial_guess=1.0,
        offset_initial_guess=0.0,
    )

    assert specs[0].scale_fixed is True
    assert specs[0].scale_fixed_value == 1.0

    assert specs[0].offset_fixed is True
    assert specs[0].offset_fixed_value == 0.0


def test_build_shared_species_observable_specs_rejects_no_columns():
    with pytest.raises(ValueError):
        build_shared_species_observable_specs(
            signal_columns=[],
            species="A",
        )


def test_build_signal_weights_from_columns():
    weights = build_signal_weights_from_columns(
        signal_columns=["A23_HN", "G45_HN"],
        default_weight=2.0,
    )

    assert weights == {
        "A23_HN": 2.0,
        "G45_HN": 2.0,
    }


def test_build_signal_weights_rejects_nonpositive_weight():
    with pytest.raises(ValueError):
        build_signal_weights_from_columns(
            signal_columns=["A23_HN"],
            default_weight=0.0,
        )


def test_fit_global_observable_model_recovers_shared_rate_constant():
    model = build_model_spec("A>B")

    true_k = 0.4

    timepoints = np.linspace(0.0, 8.0, 50)
    a_values = np.exp(-true_k * timepoints)

    peak_1 = 2.0 * a_values + 0.10
    peak_2 = 1.5 * a_values + 0.20
    peak_3 = 0.8 * a_values + 0.05

    dataset = Dataset(
        raw_dataframe=pd.DataFrame(
            {
                "time": timepoints,
                "A23_HN": peak_1,
                "G45_HN": peak_2,
                "L78_HN": peak_3,
            }
        ),
        time_column="time",
        signal_columns=["A23_HN", "G45_HN", "L78_HN"],
    )

    parameter_specs = [
        ParameterSpec(
            name="k1f",
            initial_guess=0.1,
            lower_bound=0.001,
            upper_bound=10.0,
        )
    ]

    initial_condition_specs = [
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

    output = fit_global_observable_model(
        model=model,
        dataset=dataset,
        parameter_specs=parameter_specs,
        initial_condition_specs=initial_condition_specs,
        observed_species="A",
        settings=FitSettings(
            species_mapping={},
            rtol=1e-9,
            atol=1e-11,
        ),
        scale_initial_guess=1.0,
        scale_lower_bound=0.0,
        scale_upper_bound=5.0,
        offset_initial_guess=0.0,
        offset_lower_bound=-1.0,
        offset_upper_bound=1.0,
    )

    result = output.fit_result

    assert result.success

    assert result.fitted_parameters["k1f"] == pytest.approx(
        true_k,
        rel=1e-2,
    )

    assert result.fitted_observables is not None

    assert result.fitted_observables["A23_HN"]["scale"] == pytest.approx(
        2.0,
        rel=1e-2,
    )
    assert result.fitted_observables["A23_HN"]["offset"] == pytest.approx(
        0.10,
        abs=1e-2,
    )

    assert result.fitted_observables["G45_HN"]["scale"] == pytest.approx(
        1.5,
        rel=1e-2,
    )
    assert result.fitted_observables["G45_HN"]["offset"] == pytest.approx(
        0.20,
        abs=1e-2,
    )

    assert result.fitted_observables["L78_HN"]["scale"] == pytest.approx(
        0.8,
        rel=1e-2,
    )
    assert result.fitted_observables["L78_HN"]["offset"] == pytest.approx(
        0.05,
        abs=1e-2,
    )

    # One global kinetic parameter + 3 scale values + 3 offset values.
    assert result.statistics["n_parameters"] == 7.0

    assert len(output.observable_specs) == 3


def test_fit_global_observable_model_rejects_unknown_observed_species():
    model = build_model_spec("A>B")

    dataset = Dataset(
        raw_dataframe=pd.DataFrame(
            {
                "time": [0.0, 1.0, 2.0],
                "A23_HN": [1.0, 0.8, 0.6],
            }
        ),
        time_column="time",
        signal_columns=["A23_HN"],
    )

    parameter_specs = [
        ParameterSpec(
            name="k1f",
            initial_guess=0.1,
            lower_bound=0.001,
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

    with pytest.raises(ValueError):
        fit_global_observable_model(
            model=model,
            dataset=dataset,
            parameter_specs=parameter_specs,
            initial_condition_specs=initial_condition_specs,
            observed_species="missing",
        )

        def test_read_wide_observable_dataset_with_filtering(tmp_path):
            data_path = tmp_path / "hsqc.csv"

            dataframe = pd.DataFrame(
                {
                    "time": [0.0, 1.0, 2.0, 3.0],
                    "good_peak": [1.0, np.nan, 0.5, 0.25],
                    "too_missing": [1.0, np.nan, np.nan, np.nan],
                    "flat_peak": [1.0, 1.0, 1.0, 1.0],
                    "metadata": ["a", "b", "c", "d"],
                }
            )

            dataframe.to_csv(data_path, index=False)

            dataset, filtering_result = read_wide_observable_dataset_with_filtering(
                file_path=data_path,
                time_column="time",
                exclude_columns=["metadata"],
                max_missing_fraction=0.25,
                min_dynamic_range=0.1,
                interpolate_missing=True,
            )

            assert dataset.signal_columns == ["good_peak"]
            assert filtering_result.kept_columns == ["good_peak"]
            assert set(filtering_result.removed_columns) == {"too_missing", "flat_peak"}

            assert not dataset.raw_dataframe["good_peak"].isna().any()
