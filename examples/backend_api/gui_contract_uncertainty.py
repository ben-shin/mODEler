from __future__ import annotations

from gui_contract_common import (
    base_single_species_fit_config,
    write_json_payload,
)

from odefit.api.backend import (
    bootstrap_global_observables_from_config,
    profile_likelihood_global_observables_from_config,
)
from odefit.api.serialization import backend_output_payload


def main() -> None:
    bootstrap_config = base_single_species_fit_config()
    bootstrap_config.update(
        {
            "n_bootstrap": 3,
            "n_workers": 1,
            "random_seed": 123,
            "confidence_level": 0.95,
            "show_progress": False,
            "no_plots": True,
            "output_dir": "examples/backend_api/outputs/gui_contract_bootstrap",
        }
    )

    bootstrap_output = bootstrap_global_observables_from_config(
        bootstrap_config
    )

    bootstrap_payload = backend_output_payload(
        bootstrap_output,
        workflow="bootstrap",
        max_rows=10,
    )

    bootstrap_path = write_json_payload(
        bootstrap_payload,
        "gui_contract_bootstrap_payload.json",
    )

    profile_config = base_single_species_fit_config()
    profile_config.update(
        {
            "profile_parameters": ["k1f"],
            "profile_n_points": 3,
            "profile_span_factor": 3.0,
            "profile_log_space": True,
            "show_progress": False,
            "no_plots": True,
            "output_dir": "examples/backend_api/outputs/gui_contract_profile_likelihood",
        }
    )

    profile_output = profile_likelihood_global_observables_from_config(
        profile_config
    )

    profile_payload = backend_output_payload(
        profile_output,
        workflow="profile_likelihood",
        max_rows=10,
    )

    profile_path = write_json_payload(
        profile_payload,
        "gui_contract_profile_likelihood_payload.json",
    )

    print(f"Wrote bootstrap payload to: {bootstrap_path}")
    print(f"Wrote profile likelihood payload to: {profile_path}")


if __name__ == "__main__":
    main()
