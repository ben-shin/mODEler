from __future__ import annotations

from gui_contract_common import (
    base_multispecies_fit_config,
    write_json_payload,
)

from odefit.api.backend import fit_global_observables_from_config
from odefit.api.serialization import backend_output_payload


def main() -> None:
    config = base_multispecies_fit_config()

    output = fit_global_observables_from_config(config)

    payload = backend_output_payload(
        output,
        workflow="fit",
        max_rows=10,
    )

    output_path = write_json_payload(
        payload,
        "gui_contract_multispecies_fit_payload.json",
    )

    print(f"Wrote multispecies fit payload to: {output_path}")


if __name__ == "__main__":
    main()
