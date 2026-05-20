from __future__ import annotations

import json
from pathlib import Path

from odefit.api.backend import fit_global_observables_from_config
from odefit.api.project_io import (
    attach_result_payload,
    create_project,
    load_project,
    project_to_config,
    save_project,
    summarize_project,
)
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

    project = create_project(
        name="Example GUI Project",
        description="Example saved mODEler GUI project.",
        workflow="fit",
        config=config,
        ui_state={
            "active_tab": "fit",
            "selected_time_column": "time",
            "selected_signal_columns": None,
        },
        notes=[
            "Created by examples/backend_api/demo_project_io.py",
        ],
        tags=[
            "hsqc",
            "variable-projection",
            "gui-alpha",
        ],
    )

    output = fit_global_observables_from_config(
        project_to_config(project)
    )

    payload = backend_output_payload(
        output,
        workflow="fit",
        max_rows=10,
    )

    attach_result_payload(
        project=project,
        result_payload=payload,
    )

    project_path = Path("examples/backend_api/example_project.modeler.json")

    save_project(
        project=project,
        file_path=project_path,
    )

    loaded_project = load_project(project_path)

    summary = summarize_project(loaded_project)

    summary_path = Path("examples/backend_api/example_project_summary.json")

    with summary_path.open("w") as handle:
        json.dump(
            summary,
            handle,
            indent=2,
        )

    print(f"Wrote project file to: {project_path}")
    print(f"Wrote project summary to: {summary_path}")


if __name__ == "__main__":
    main()
