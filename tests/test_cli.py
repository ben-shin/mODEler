import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from odefit.cli import (
    build_default_species_mapping,
    build_initial_condition_specs,
    build_parameter_specs,
    main,
    parse_mapping_entries,
    parse_parameter_entries,
    parse_signal_weight_entries,
)
from odefit.model.model_spec import build_model_spec
from odefit.simulation.simulation_result import SimulationResult


def test_parse_mapping_entries():
    mapping = parse_mapping_entries(
        [
            "amide:A",
            "signal_b:B",
        ]
    )

    assert mapping == {
        "amide": "A",
        "signal_b": "B",
    }


def test_parse_mapping_entries_rejects_bad_format():
    with pytest.raises(ValueError):
        parse_mapping_entries(["amide-A"])


def test_build_default_species_mapping_single_signal_maps_to_a():
    model = build_model_spec("A>B")

    mapping = build_default_species_mapping(
        signal_columns=["amide"],
        model=model,
    )

    assert mapping == {
        "amide": "A",
    }


def test_build_default_species_mapping_identical_names():
    model = build_model_spec("A>B")

    mapping = build_default_species_mapping(
        signal_columns=["A", "B"],
        model=model,
    )

    assert mapping == {
        "A": "A",
        "B": "B",
    }


def test_build_default_species_mapping_rejects_unknown_column():
    model = build_model_spec("A>B")

    with pytest.raises(ValueError):
        build_default_species_mapping(
            signal_columns=["unknown_a", "unknown_b"],
            model=model,
        )


def test_parse_parameter_entries():
    parsed = parse_parameter_entries(
        [
            "k1f:0.01:0:10",
            "k1r:0.02:0:20",
        ]
    )

    assert parsed == {
        "k1f": (0.01, 0.0, 10.0),
        "k1r": (0.02, 0.0, 20.0),
    }


def test_build_parameter_specs_uses_defaults_and_overrides():
    model = build_model_spec("A-B")

    specs = build_parameter_specs(
        model=model,
        parameter_entries=["k1f:0.01:0:10"],
        default_guess=0.2,
        default_lower=0.0,
        default_upper=100.0,
    )

    by_name = {spec.name: spec for spec in specs}

    assert by_name["k1f"].initial_guess == 0.01
    assert by_name["k1f"].upper_bound == 10.0

    assert by_name["k1r"].initial_guess == 0.2
    assert by_name["k1r"].upper_bound == 100.0


def test_build_parameter_specs_rejects_unknown_parameter():
    model = build_model_spec("A>B")

    with pytest.raises(ValueError):
        build_parameter_specs(
            model=model,
            parameter_entries=["missing:0.1:0:10"],
        )


def test_build_initial_condition_specs_defaults():
    model = build_model_spec("A>B")

    specs = build_initial_condition_specs(model)

    by_species = {spec.species: spec for spec in specs}

    assert by_species["A"].fixed is True
    assert by_species["A"].fixed_value == 1.0

    assert by_species["B"].fixed is True
    assert by_species["B"].fixed_value == 0.0


def test_build_initial_condition_specs_override_fit_mode():
    model = build_model_spec("A>B")

    specs = build_initial_condition_specs(
        model=model,
        initial_entries=["A:0.8:fit:0:2"],
    )

    by_species = {spec.species: spec for spec in specs}

    assert by_species["A"].fixed is False
    assert by_species["A"].initial_guess == 0.8
    assert by_species["A"].lower_bound == 0.0
    assert by_species["A"].upper_bound == 2.0
    assert by_species["A"].fixed_value is None


def test_parse_signal_weight_entries():
    weights = parse_signal_weight_entries(["A:2.0", "B:0.5"])

    assert weights == {
        "A": 2.0,
        "B": 0.5,
    }


def test_generate_odes_command_writes_output(tmp_path):
    model_path = tmp_path / "model.txt"
    output_path = tmp_path / "odes.txt"

    model_path.write_text("A>B")

    main(
        [
            "generate-odes",
            "--model",
            str(model_path),
            "--output",
            str(output_path),
        ]
    )

    assert output_path.exists()

    text = output_path.read_text()

    assert "dA/dt" in text
    assert "dB/dt" in text


def test_fit_command_writes_output_bundle(tmp_path):
    model_path = tmp_path / "model.txt"
    data_path = tmp_path / "data.csv"
    output_dir = tmp_path / "fit_output"

    model_path.write_text("A>B")

    dataframe = pd.DataFrame(
        {
            "time": [0.0, 1.0, 2.0, 3.0],
            "A": [1.0, 0.6, 0.35, 0.2],
            "B": [0.0, 0.4, 0.65, 0.8],
        }
    )

    dataframe.to_csv(data_path, index=False)

    main(
        [
            "fit",
            "--model",
            str(model_path),
            "--data",
            str(data_path),
            "--time-column",
            "time",
            "--signal-columns",
            "A",
            "B",
            "--mapping",
            "A:A",
            "--mapping",
            "B:B",
            "--parameter",
            "k1f:0.2:0:10",
            "--initial",
            "A:1.0:fixed:0:10",
            "--initial",
            "B:0.0:fixed:0:10",
            "--output-dir",
            str(output_dir),
            "--no-plots",
        ]
    )

    assert output_dir.exists()
    assert (output_dir / "fit_statistics.csv").exists()
    assert (output_dir / "fitted_parameters.csv").exists()
    assert (output_dir / "fitted_initial_conditions.csv").exists()
    assert (output_dir / "simulated_curves.csv").exists()
    assert (output_dir / "residuals.csv").exists()
    assert (output_dir / "optimizer_diagnostics.csv").exists()


def test_parse_mapping_entries_accepts_dict():
    mapping = parse_mapping_entries(
        {
            "amide": "A",
            "signal_b": "B",
        }
    )

    assert mapping == {
        "amide": "A",
        "signal_b": "B",
    }


def test_parse_parameter_entries_accepts_dict():
    parsed = parse_parameter_entries(
        {
            "k1f": {
                "initial_guess": 0.01,
                "lower_bound": 0.0,
                "upper_bound": 10.0,
            }
        }
    )

    assert parsed == {
        "k1f": (0.01, 0.0, 10.0),
    }


def test_parse_signal_weight_entries_accepts_dict():
    weights = parse_signal_weight_entries(
        {
            "A": 2.0,
            "B": 0.5,
        }
    )

    assert weights == {
        "A": 2.0,
        "B": 0.5,
    }


def test_fit_command_with_config_writes_output_bundle(tmp_path):
    model_path = tmp_path / "model.txt"
    data_path = tmp_path / "data.csv"
    output_dir = tmp_path / "fit_output"
    config_path = tmp_path / "fit_config.json"

    model_path.write_text("A>B")

    dataframe = pd.DataFrame(
        {
            "time": [0.0, 1.0, 2.0, 3.0],
            "A": [1.0, 0.6, 0.35, 0.2],
            "B": [0.0, 0.4, 0.65, 0.8],
        }
    )

    dataframe.to_csv(data_path, index=False)

    config = {
        "model": str(model_path),
        "data": str(data_path),
        "time_column": "time",
        "signal_columns": ["A", "B"],
        "mapping": {
            "A": "A",
            "B": "B",
        },
        "parameters": {
            "k1f": {
                "initial_guess": 0.2,
                "lower_bound": 0.0,
                "upper_bound": 10.0,
            }
        },
        "initial_conditions": {
            "A": {
                "value": 1.0,
                "mode": "fixed",
                "lower_bound": 0.0,
                "upper_bound": 10.0,
            },
            "B": {
                "value": 0.0,
                "mode": "fixed",
                "lower_bound": 0.0,
                "upper_bound": 10.0,
            },
        },
        "signal_weights": {
            "A": 1.0,
            "B": 1.0,
        },
        "method": "trf",
        "loss": "linear",
        "rtol": 1e-6,
        "atol": 1e-9,
        "max_nfev": 1000,
        "output_dir": str(output_dir),
        "no_plots": True,
    }

    config_path.write_text(json.dumps(config))

    main(
        [
            "fit",
            "--config",
            str(config_path),
        ]
    )

    assert output_dir.exists()
    assert (output_dir / "fit_statistics.csv").exists()
    assert (output_dir / "optimizer_diagnostics.csv").exists()
    assert (output_dir / "fit_diagnostics.csv").exists()
    assert (output_dir / "fitted_parameters.csv").exists()
    assert (output_dir / "fitted_initial_conditions.csv").exists()
    assert (output_dir / "simulated_curves.csv").exists()
    assert (output_dir / "residuals.csv").exists()


def test_multistart_command_with_config_writes_outputs(tmp_path):
    model_path = tmp_path / "model.txt"
    data_path = tmp_path / "data.csv"
    output_dir = tmp_path / "multistart_output"
    config_path = tmp_path / "multistart_config.json"

    model_path.write_text("A>B")

    dataframe = pd.DataFrame(
        {
            "time": [0.0, 1.0, 2.0, 3.0, 4.0],
            "A": [1.0, 0.6, 0.35, 0.2, 0.12],
            "B": [0.0, 0.4, 0.65, 0.8, 0.88],
        }
    )

    dataframe.to_csv(data_path, index=False)

    config = {
        "model": str(model_path),
        "data": str(data_path),
        "time_column": "time",
        "signal_columns": ["A", "B"],
        "mapping": {
            "A": "A",
            "B": "B",
        },
        "parameters": {
            "k1f": {
                "initial_guess": 0.2,
                "lower_bound": 0.001,
                "upper_bound": 10.0,
            }
        },
        "initial_conditions": {
            "A": {
                "value": 1.0,
                "mode": "fixed",
                "lower_bound": 0.0,
                "upper_bound": 10.0,
            },
            "B": {
                "value": 0.0,
                "mode": "fixed",
                "lower_bound": 0.0,
                "upper_bound": 10.0,
            },
        },
        "signal_weights": {
            "A": 1.0,
            "B": 1.0,
        },
        "method": "trf",
        "loss": "linear",
        "rtol": 1e-6,
        "atol": 1e-9,
        "max_nfev": 1000,
        "output_dir": str(output_dir),
        "no_plots": True,
        "n_starts": 3,
        "n_workers": 1,
        "random_seed": 1,
        "sort_by": "aic",
        "log_uniform": True,
    }

    config_path.write_text(json.dumps(config))

    main(
        [
            "multistart",
            "--config",
            str(config_path),
        ]
    )

    assert output_dir.exists()

    assert (output_dir / "multistart_comparison.csv").exists()
    assert (output_dir / "multistart_starting_parameters.csv").exists()

    best_fit_dir = output_dir / "best_fit"

    assert best_fit_dir.exists()
    assert (best_fit_dir / "fit_statistics.csv").exists()
    assert (best_fit_dir / "optimizer_diagnostics.csv").exists()
    assert (best_fit_dir / "fit_diagnostics.csv").exists()
    assert (best_fit_dir / "fitted_parameters.csv").exists()
    assert (best_fit_dir / "fitted_initial_conditions.csv").exists()
    assert (best_fit_dir / "simulated_curves.csv").exists()
    assert (best_fit_dir / "residuals.csv").exists()


from odefit.cli import (
    build_simulation_timepoints,
    parse_float_value_entries,
    write_simulation_csv,
)
from odefit.simulation.simulation_result import SimulationResult


def test_parse_float_value_entries_from_cli_list():
    values = parse_float_value_entries(
        [
            "k1f:0.5",
            "k1r:0.1",
        ]
    )

    assert values == {
        "k1f": 0.5,
        "k1r": 0.1,
    }


def test_parse_float_value_entries_from_dict():
    values = parse_float_value_entries(
        {
            "k1f": 0.5,
            "k1r": 0.1,
        }
    )

    assert values == {
        "k1f": 0.5,
        "k1r": 0.1,
    }


def test_parse_float_value_entries_from_nested_dict():
    values = parse_float_value_entries(
        {
            "A": {
                "value": 1.0,
            },
            "B": {
                "initial_guess": 0.0,
            },
        }
    )

    assert values == {
        "A": 1.0,
        "B": 0.0,
    }


def test_build_simulation_timepoints_from_explicit_values():
    timepoints = build_simulation_timepoints(
        timepoints=["0", "1", "2"],
    )

    assert np.allclose(timepoints, [0.0, 1.0, 2.0])


def test_build_simulation_timepoints_from_range():
    timepoints = build_simulation_timepoints(
        time_start=0.0,
        time_end=10.0,
        num_points=6,
    )

    assert np.allclose(timepoints, [0.0, 2.0, 4.0, 6.0, 8.0, 10.0])


def test_write_simulation_csv(tmp_path):
    result = SimulationResult(
        timepoints=np.array([0.0, 1.0]),
        species=["A", "B"],
        values=np.array(
            [
                [1.0, 0.0],
                [0.5, 0.5],
            ]
        ),
    )

    output_csv = tmp_path / "simulation.csv"

    written_path = write_simulation_csv(
        simulation_result=result,
        output_csv=output_csv,
    )

    assert written_path == output_csv
    assert output_csv.exists()

    dataframe = pd.read_csv(output_csv)

    assert list(dataframe.columns) == ["time", "A", "B"]
    assert list(dataframe["A"]) == [1.0, 0.5]


def test_simulate_command_writes_csv(tmp_path):
    model_path = tmp_path / "model.txt"
    output_csv = tmp_path / "simulation.csv"

    model_path.write_text("A>B")

    main(
        [
            "simulate",
            "--model",
            str(model_path),
            "--parameter-value",
            "k1f:0.5",
            "--initial-value",
            "A:1.0",
            "--initial-value",
            "B:0.0",
            "--time-start",
            "0",
            "--time-end",
            "5",
            "--num-points",
            "6",
            "--output-csv",
            str(output_csv),
        ]
    )

    assert output_csv.exists()

    dataframe = pd.read_csv(output_csv)

    assert list(dataframe.columns) == ["time", "A", "B"]
    assert len(dataframe) == 6
    assert dataframe["A"].iloc[0] == 1.0
    assert dataframe["B"].iloc[0] == 0.0


def test_simulate_command_with_config_writes_csv(tmp_path):
    model_path = tmp_path / "model.txt"
    output_csv = tmp_path / "simulation_from_config.csv"
    config_path = tmp_path / "simulation_config.json"

    model_path.write_text("A>B")

    config = {
        "model": str(model_path),
        "parameter_values": {"k1f": 0.5},
        "initial_values": {"A": 1.0, "B": 0.0},
        "time_start": 0.0,
        "time_end": 5.0,
        "num_points": 6,
        "method": "LSODA",
        "rtol": 1e-6,
        "atol": 1e-9,
        "output_csv": str(output_csv),
    }

    config_path.write_text(json.dumps(config))

    main(
        [
            "simulate",
            "--config",
            str(config_path),
        ]
    )

    assert output_csv.exists()

    dataframe = pd.read_csv(output_csv)

    assert list(dataframe.columns) == ["time", "A", "B"]
    assert len(dataframe) == 6


def test_parse_float_value_entries_from_cli_list():
    values = parse_float_value_entries(
        [
            "k1f:0.5",
            "k1r:0.1",
        ]
    )

    assert values == {
        "k1f": 0.5,
        "k1r": 0.1,
    }


def test_parse_float_value_entries_from_dict():
    values = parse_float_value_entries(
        {
            "k1f": 0.5,
            "k1r": 0.1,
        }
    )

    assert values == {
        "k1f": 0.5,
        "k1r": 0.1,
    }


def test_parse_float_value_entries_from_nested_dict():
    values = parse_float_value_entries(
        {
            "A": {
                "value": 1.0,
            },
            "B": {
                "initial_guess": 0.0,
            },
        }
    )

    assert values == {
        "A": 1.0,
        "B": 0.0,
    }


def test_build_simulation_timepoints_from_explicit_values():
    timepoints = build_simulation_timepoints(
        timepoints=["0", "1", "2"],
    )

    assert np.allclose(timepoints, [0.0, 1.0, 2.0])


def test_build_simulation_timepoints_from_range():
    timepoints = build_simulation_timepoints(
        time_start=0.0,
        time_end=10.0,
        num_points=6,
    )

    assert np.allclose(timepoints, [0.0, 2.0, 4.0, 6.0, 8.0, 10.0])


def test_write_simulation_csv(tmp_path):
    result = SimulationResult(
        timepoints=np.array([0.0, 1.0]),
        species=["A", "B"],
        values=np.array(
            [
                [1.0, 0.0],
                [0.5, 0.5],
            ]
        ),
    )

    output_csv = tmp_path / "simulation.csv"

    written_path = write_simulation_csv(
        simulation_result=result,
        output_csv=output_csv,
    )

    assert written_path == output_csv
    assert output_csv.exists()

    dataframe = pd.read_csv(output_csv)

    assert list(dataframe.columns) == ["time", "A", "B"]
    assert list(dataframe["A"]) == [1.0, 0.5]


def test_simulate_command_writes_csv(tmp_path):
    model_path = tmp_path / "model.txt"
    output_csv = tmp_path / "simulation.csv"

    model_path.write_text("A>B")

    main(
        [
            "simulate",
            "--model",
            str(model_path),
            "--parameter-value",
            "k1f:0.5",
            "--initial-value",
            "A:1.0",
            "--initial-value",
            "B:0.0",
            "--time-start",
            "0",
            "--time-end",
            "5",
            "--num-points",
            "6",
            "--output-csv",
            str(output_csv),
        ]
    )

    assert output_csv.exists()

    dataframe = pd.read_csv(output_csv)

    assert list(dataframe.columns) == ["time", "A", "B"]
    assert len(dataframe) == 6
    assert dataframe["A"].iloc[0] == 1.0
    assert dataframe["B"].iloc[0] == 0.0


def test_simulate_command_with_config_writes_csv(tmp_path):
    model_path = tmp_path / "model.txt"
    output_csv = tmp_path / "simulation_from_config.csv"
    config_path = tmp_path / "simulation_config.json"

    model_path.write_text("A>B")

    config = {
        "model": str(model_path),
        "parameter_values": {
            "k1f": 0.5,
        },
        "initial_values": {
            "A": 1.0,
            "B": 0.0,
        },
        "time_start": 0.0,
        "time_end": 5.0,
        "num_points": 6,
        "method": "LSODA",
        "rtol": 1e-6,
        "atol": 1e-9,
        "output_csv": str(output_csv),
    }

    config_path.write_text(json.dumps(config))

    main(
        [
            "simulate",
            "--config",
            str(config_path),
        ]
    )

    assert output_csv.exists()

    dataframe = pd.read_csv(output_csv)

    assert list(dataframe.columns) == ["time", "A", "B"]
    assert len(dataframe) == 6


def test_fit_global_observables_command_with_config_writes_output_bundle(tmp_path):
    model_path = tmp_path / "model.txt"
    data_path = tmp_path / "hsqc.csv"
    output_dir = tmp_path / "global_observable_fit"
    config_path = tmp_path / "global_observable_config.json"

    model_path.write_text("A>B")

    true_k = 0.4
    timepoints = np.linspace(0.0, 8.0, 30)
    a_values = np.exp(-true_k * timepoints)

    dataframe = pd.DataFrame(
        {
            "time": timepoints,
            "A23_HN": 2.0 * a_values + 0.10,
            "G45_HN": 1.5 * a_values + 0.20,
            "L78_HN": 0.8 * a_values + 0.05,
        }
    )

    dataframe.to_csv(data_path, index=False)

    config = {
        "model": str(model_path),
        "data": str(data_path),
        "time_column": "time",
        "observed_species": "A",
        "parameters": {
            "k1f": {
                "initial_guess": 0.1,
                "lower_bound": 0.001,
                "upper_bound": 10.0,
            }
        },
        "initial_conditions": {
            "A": {
                "value": 1.0,
                "mode": "fixed",
                "lower_bound": 0.0,
                "upper_bound": 2.0,
            },
            "B": {
                "value": 0.0,
                "mode": "fixed",
                "lower_bound": 0.0,
                "upper_bound": 2.0,
            },
        },
        "fit_scale": True,
        "fit_offset": True,
        "scale_initial_guess": 1.0,
        "scale_lower_bound": 0.0,
        "scale_upper_bound": 5.0,
        "offset_initial_guess": 0.0,
        "offset_lower_bound": -1.0,
        "offset_upper_bound": 1.0,
        "method": "trf",
        "loss": "linear",
        "rtol": 1e-8,
        "atol": 1e-10,
        "max_nfev": 2000,
        "max_missing_fraction": 0.25,
        "min_initial_intensity": None,
        "initial_points": 1,
        "min_dynamic_range": None,
        "interpolate_missing": True,
        "output_dir": str(output_dir),
        "no_plots": True,
    }

    config_path.write_text(json.dumps(config))

    main(
        [
            "fit-global-observables",
            "--config",
            str(config_path),
        ]
    )

    assert output_dir.exists()

    assert (output_dir / "fit_statistics.csv").exists()
    assert (output_dir / "optimizer_diagnostics.csv").exists()
    assert (output_dir / "fit_diagnostics.csv").exists()
    assert (output_dir / "fitted_parameters.csv").exists()
    assert (output_dir / "fitted_initial_conditions.csv").exists()
    assert (output_dir / "fitted_observables.csv").exists()
    assert (output_dir / "simulated_curves.csv").exists()
    assert (output_dir / "residuals.csv").exists()
    assert (output_dir / "peak_filtering.csv").exists()

    fitted_parameters = pd.read_csv(output_dir / "fitted_parameters.csv")

    k1f_row = fitted_parameters[fitted_parameters["parameter"] == "k1f"].iloc[0]

    assert k1f_row["fitted_value"] == pytest.approx(true_k, rel=1e-2)

    fitted_observables = pd.read_csv(output_dir / "fitted_observables.csv")

    assert set(fitted_observables["data_column"]) == {
        "A23_HN",
        "G45_HN",
        "L78_HN",
    }

    residuals = pd.read_csv(output_dir / "residuals.csv")

    assert "A23_HN_observed" in residuals.columns
    assert "A23_HN_fit" in residuals.columns
    assert "A23_HN_residual" in residuals.columns


def test_multistart_global_observables_command_with_config_writes_outputs(tmp_path):
    model_path = tmp_path / "model.txt"
    data_path = tmp_path / "hsqc.csv"
    output_dir = tmp_path / "global_observable_multistart"
    config_path = tmp_path / "global_observable_multistart_config.json"

    model_path.write_text("A>B")

    true_k = 0.4
    timepoints = np.linspace(0.0, 8.0, 30)
    a_values = np.exp(-true_k * timepoints)

    dataframe = pd.DataFrame(
        {
            "time": timepoints,
            "A23_HN": 2.0 * a_values + 0.10,
            "G45_HN": 1.5 * a_values + 0.20,
            "L78_HN": 0.8 * a_values + 0.05,
        }
    )

    dataframe.to_csv(data_path, index=False)

    config = {
        "model": str(model_path),
        "data": str(data_path),
        "time_column": "time",
        "observed_species": "A",
        "parameters": {
            "k1f": {
                "initial_guess": 0.1,
                "lower_bound": 0.001,
                "upper_bound": 10.0,
            }
        },
        "initial_conditions": {
            "A": {
                "value": 1.0,
                "mode": "fixed",
                "lower_bound": 0.0,
                "upper_bound": 2.0,
            },
            "B": {
                "value": 0.0,
                "mode": "fixed",
                "lower_bound": 0.0,
                "upper_bound": 2.0,
            },
        },
        "fit_scale": True,
        "fit_offset": True,
        "scale_initial_guess": 1.0,
        "scale_lower_bound": 0.0,
        "scale_upper_bound": 5.0,
        "offset_initial_guess": 0.0,
        "offset_lower_bound": -1.0,
        "offset_upper_bound": 1.0,
        "method": "trf",
        "loss": "linear",
        "rtol": 1e-8,
        "atol": 1e-10,
        "max_nfev": 2000,
        "max_missing_fraction": 0.25,
        "min_initial_intensity": None,
        "initial_points": 1,
        "min_dynamic_range": None,
        "interpolate_missing": True,
        "output_dir": str(output_dir),
        "no_plots": True,
        "n_starts": 3,
        "n_workers": 1,
        "random_seed": 1,
        "sort_by": "aic",
        "log_uniform": True,
    }

    config_path.write_text(json.dumps(config))

    main(
        [
            "multistart-global-observables",
            "--config",
            str(config_path),
        ]
    )

    assert output_dir.exists()

    assert (output_dir / "global_observable_multistart_comparison.csv").exists()
    assert (
        output_dir / "global_observable_multistart_starting_parameters.csv"
    ).exists()

    best_fit_dir = output_dir / "best_fit"
    assert (output_dir / "peak_filtering.csv").exists()

    assert best_fit_dir.exists()

    assert (best_fit_dir / "fit_statistics.csv").exists()
    assert (best_fit_dir / "optimizer_diagnostics.csv").exists()
    assert (best_fit_dir / "fit_diagnostics.csv").exists()
    assert (best_fit_dir / "fitted_parameters.csv").exists()
    assert (best_fit_dir / "fitted_initial_conditions.csv").exists()
    assert (best_fit_dir / "fitted_observables.csv").exists()
    assert (best_fit_dir / "simulated_curves.csv").exists()
    assert (best_fit_dir / "residuals.csv").exists()

    fitted_parameters = pd.read_csv(best_fit_dir / "fitted_parameters.csv")

    k1f_row = fitted_parameters[fitted_parameters["parameter"] == "k1f"].iloc[0]

    assert k1f_row["fitted_value"] == pytest.approx(true_k, rel=1e-2)

    fitted_observables = pd.read_csv(best_fit_dir / "fitted_observables.csv")

    assert set(fitted_observables["data_column"]) == {
        "A23_HN",
        "G45_HN",
        "L78_HN",
    }

    residuals = pd.read_csv(best_fit_dir / "residuals.csv")

    assert "A23_HN_observed" in residuals.columns
    assert "A23_HN_fit" in residuals.columns
    assert "A23_HN_residual" in residuals.columns
