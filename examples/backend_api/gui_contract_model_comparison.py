from __future__ import annotations

from gui_contract_common import (
    base_single_species_model_comparison_config,
    write_json_payload,
)

from odefit.api.backend import compare_global_observables_from_config
from odefit.api.serialization import backend_output_payload


def main() -> None:
    config = base_single_species_model_comparison_config()

    output = compare_global_observables_from_config(config)

    payload = backend_output_payload(
        output,
        workflow="model_comparison",
        max_rows=20,
    )

    output_path = write_json_payload(
        payload,
        "gui_contract_model_comparison_payload.json",
    )

    print(f"Wrote model comparison payload to: {output_path}")


if __name__ == "__main__":
    main()
