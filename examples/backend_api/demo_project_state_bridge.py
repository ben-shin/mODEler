from __future__ import annotations

import json
from pathlib import Path

from odefit.api.project_state_bridge import (
    project_state_to_gui_project_payload,
)
from odefit.project.project_io import load_project_state


def main() -> None:
    project_path = Path("examples/backend_api/example_project.modeler.json")

    project_state = load_project_state(project_path)

    payload = project_state_to_gui_project_payload(
        project_state,
        workflow="fit",
        use_variable_projection=True,
    )

    output_path = Path("examples/backend_api/project_state_gui_payload.json")

    with output_path.open("w") as handle:
        json.dump(payload, handle, indent=2)

    print(f"Wrote GUI project-state payload to: {output_path}")


if __name__ == "__main__":
    main()
