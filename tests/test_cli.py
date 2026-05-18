from pathlib import Path

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
