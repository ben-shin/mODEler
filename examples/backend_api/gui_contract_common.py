from __future__ import annotations

import json
from pathlib import Path
from typing import Any


EXAMPLE_OUTPUT_DIR = Path("examples/backend_api/outputs")


def ensure_output_dir() -> Path:
    EXAMPLE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return EXAMPLE_OUTPUT_DIR


def write_json_payload(
    payload: dict[str, Any],
    filename: str,
) -> Path:
    output_dir = ensure_output_dir()
    output_path = output_dir / filename

    with output_path.open("w") as handle:
        json.dump(
            payload,
            handle,
            indent=2,
        )

    return output_path


def base_single_species_fit_config() -> dict[str, Any]:
    return {
        "model": "examples/configs/model_first_order.txt",
        "data": "examples/configs/example_hsqc_peaks.csv",
        "time_column": "time",
        "signal_columns": None,
        "exclude_columns": None,
        "observed_species": "A",
        "output_dir": "examples/backend_api/outputs/gui_contract_fit",

        "parameters": {
            "k1f": {
                "initial_guess": 0.1,
                "lower_bound": 0.000001,
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

        "use_variable_projection": True,
        "fit_scale": True,
        "fit_offset": True,

        "method": "trf",
        "loss": "linear",
        "max_nfev": 100,
        "rtol": 0.000001,
        "atol": 0.000000001,

        "variable_projection_backend": "numpy",
        "variable_projection_method": "LSODA",

        "max_missing_fraction": 0.0,
        "min_initial_intensity": None,
        "initial_points": 1,
        "min_dynamic_range": None,
        "interpolate_missing": True,

        "show_progress": False,
        "no_plots": True,
    }


def base_multispecies_fit_config() -> dict[str, Any]:
    config = base_single_species_fit_config()

    config.update(
        {
            "observed_species": ["A", "B"],
            "use_variable_projection": False,
            "use_multispecies_variable_projection": True,
            "fit_offset": True,
            "output_dir": "examples/backend_api/outputs/gui_contract_multispecies_fit",
        }
    )

    config.pop("fit_scale", None)

    return config


def base_single_species_model_comparison_config() -> dict[str, Any]:
    return {
        "model_texts": {
            "single_step": "A -> B",
            "two_step": "A -> B\nB -> C",
        },
        "data": "examples/configs/example_hsqc_peaks.csv",
        "time_column": "time",
        "signal_columns": None,
        "exclude_columns": None,
        "observed_species": "A",
        "output_dir": "examples/backend_api/outputs/gui_contract_model_comparison",

        "parameters_by_model": {
            "single_step": {
                "k1f": {
                    "initial_guess": 0.1,
                    "lower_bound": 0.000001,
                    "upper_bound": 10.0,
                }
            },
            "two_step": {
                "k1f": {
                    "initial_guess": 0.1,
                    "lower_bound": 0.000001,
                    "upper_bound": 10.0,
                },
                "k2f": {
                    "initial_guess": 0.05,
                    "lower_bound": 0.000001,
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

        "use_variable_projection": True,
        "fit_scale": True,
        "fit_offset": True,
        "sort_by": "bic",

        "method": "trf",
        "loss": "linear",
        "max_nfev": 100,
        "rtol": 0.000001,
        "atol": 0.000000001,

        "variable_projection_backend": "numpy",
        "variable_projection_method": "LSODA",

        "max_missing_fraction": 0.0,
        "min_initial_intensity": None,
        "initial_points": 1,
        "min_dynamic_range": None,
        "interpolate_missing": True,

        "show_progress": False,
    }
