from __future__ import annotations

import json
from pathlib import Path

from odefit.api.backend import fit_global_observables_from_config
from odefit.api.serialization import backend_output_payload


def main() -> None:
    config = {
        "model": "examples/configs/model_first_order.txt",
        "data": "examples/configs/example_hsqc_peaks.csv",
        "time_column": "time",
        "observed_species": "A",
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
        "max_nfev": 100,
        "show_progress": False,
    }

    output = fit_global_observables_from_config(config)

    payload = backend_output_payload(
        output,
        workflow="fit",
        max_rows=10,
    )

    output_path = Path("examples/backend_api/backend_api_fit_payload.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as handle:
        json.dump(payload, handle, indent=2)

    print(f"Wrote GUI-style backend payload to: {output_path}")


if __name__ == "__main__":
    main()
