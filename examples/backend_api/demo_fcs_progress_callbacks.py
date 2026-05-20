from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from odefit.api.fcs import compare_fcs_models_from_config
from odefit.progress import ProgressEvent


def on_progress(event: ProgressEvent) -> None:
    print(event.to_console_line())


def main() -> None:
    output_dir = Path("examples/backend_api/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    data_path = output_dir / "demo_fcs_data.csv"

    tau = np.linspace(0.0, 5.0, 20)
    signal = np.exp(-0.4 * tau)

    dataframe = pd.DataFrame(
        {
            "tau": tau,
            "trace_0": 1.2 * signal + 0.05,
            "trace_1": 0.8 * signal - 0.02,
            "trace_2": 1.8 * signal + 0.1,
        }
    )

    dataframe.to_csv(data_path, index=False)

    config = {
        "engine_name": "reference",
        "data": str(data_path),
        "time_column": "tau",
        "signal_columns": ["trace_0", "trace_1", "trace_2"],
        "observed_species": "A",
        "use_variable_projection": True,
        "fit_scale": True,
        "fit_offset": True,
        "model_texts": {
            "single_step": "A -> B",
            "two_step": "A -> B\nB -> C",
        },
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
                "A": {"value": 1.0, "mode": "fixed"},
                "B": {"value": 0.0, "mode": "fixed"},
            },
            "two_step": {
                "A": {"value": 1.0, "mode": "fixed"},
                "B": {"value": 0.0, "mode": "fixed"},
                "C": {"value": 0.0, "mode": "fixed"},
            },
        },
        "method": "trf",
        "loss": "linear",
        "max_nfev": 80,
        "rtol": 1e-6,
        "atol": 1e-9,
        "variable_projection_backend": "numpy",
        "variable_projection_method": "LSODA",
        "show_progress": False,
        "no_plots": True,
        "sort_by": "bic",
    }

    output = compare_fcs_models_from_config(
        config,
        progress_callback=on_progress,
    )

    table = output["comparison_table"]

    print("\nModel comparison table")
    print("======================")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
