from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from odefit.project.project_state import ProjectState


def _parameter_spec_to_config(spec) -> dict[str, Any]:
    return {
        "initial_guess": float(spec.initial_guess),
        "lower_bound": float(spec.lower_bound),
        "upper_bound": float(spec.upper_bound),
    }


def _initial_condition_spec_to_config(spec) -> dict[str, Any]:
    if getattr(spec, "fixed", False):
        return {
            "value": float(spec.fixed_value),
            "mode": "fixed",
        }

    return {
        "value": float(spec.initial_guess),
        "mode": "fit",
        "initial_guess": float(spec.initial_guess),
        "lower_bound": float(spec.lower_bound),
        "upper_bound": float(spec.upper_bound),
    }


def _fit_settings_to_config(fit_settings) -> dict[str, Any]:
    if fit_settings is None:
        return {}

    return {
        "method": getattr(fit_settings, "method", "trf"),
        "loss": getattr(fit_settings, "loss", "linear"),
        "max_nfev": getattr(fit_settings, "max_nfev", None),
        "rtol": getattr(fit_settings, "rtol", 1e-6),
        "atol": getattr(fit_settings, "atol", 1e-9),
        "signal_weights": getattr(fit_settings, "signal_weights", None),
    }


def _unique_in_order(values: Iterable[str]) -> list[str]:
    seen = set()
    output = []

    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)

    return output


def infer_observed_species_from_project_state(
    project_state: ProjectState,
) -> str | list[str] | None:
    observable_specs = getattr(project_state, "observable_specs", [])

    if observable_specs:
        species = _unique_in_order(
            str(spec.species)
            for spec in observable_specs
        )

        if len(species) == 1:
            return species[0]

        return species

    species_mapping = getattr(project_state, "species_mapping", None)

    if species_mapping:
        species = _unique_in_order(
            str(value)
            for value in species_mapping.values()
        )

        if len(species) == 1:
            return species[0]

        return species

    return None


def project_state_to_backend_config(
    project_state: ProjectState,
    *,
    workflow: str = "fit",
    use_variable_projection: bool = True,
    use_multispecies_variable_projection: bool = False,
    observed_species: str | list[str] | None = None,
    fit_scale: bool = True,
    fit_offset: bool = True,
    variable_projection_backend: str = "numpy",
    variable_projection_method: str = "LSODA",
    output_dir: str | None = None,
) -> dict[str, Any]:
    """
    Convert the existing internal ProjectState object into a config dictionary
    accepted by the GUI-facing backend API.

    This is intentionally conservative:
    - It preserves model text, data path, columns, parameters, initial conditions.
    - It adds variable-projection flags for modern HSQC/global observable workflows.
    - It does not mutate the original ProjectState.
    """

    if observed_species is None:
        observed_species = infer_observed_species_from_project_state(
            project_state
        )

    config: dict[str, Any] = {
        "model_text": project_state.model_text,
        "data": project_state.data_path,
        "time_column": project_state.time_column,
        "signal_columns": list(project_state.signal_columns),
        "exclude_columns": None,
        "output_dir": output_dir
        or getattr(project_state, "output_dir", None)
        or "outputs",
        "parameters": {
            spec.name: _parameter_spec_to_config(spec)
            for spec in project_state.parameter_specs
        },
        "initial_conditions": {
            spec.species: _initial_condition_spec_to_config(spec)
            for spec in project_state.initial_condition_specs
        },
        "fit_offset": fit_offset,
        "method": "trf",
        "loss": "linear",
        "max_nfev": None,
        "rtol": 1e-6,
        "atol": 1e-9,
        "variable_projection_backend": variable_projection_backend,
        "variable_projection_method": variable_projection_method,
    }

    config.update(
        _fit_settings_to_config(
            getattr(project_state, "fit_settings", None)
        )
    )

    if observed_species is not None:
        config["observed_species"] = observed_species

    if use_multispecies_variable_projection:
        config["use_multispecies_variable_projection"] = True
        config["use_variable_projection"] = False

        if isinstance(config.get("observed_species"), str):
            config["observed_species"] = [config["observed_species"]]

    elif use_variable_projection:
        config["use_variable_projection"] = True
        config["use_multispecies_variable_projection"] = False
        config["fit_scale"] = fit_scale

        if isinstance(config.get("observed_species"), list):
            if len(config["observed_species"]) == 1:
                config["observed_species"] = config["observed_species"][0]
            else:
                raise ValueError(
                    "Single-species variable projection requires one "
                    "observed species. Use "
                    "use_multispecies_variable_projection=True for multiple "
                    "observed species."
                )

    if workflow == "bootstrap":
        config.setdefault("n_bootstrap", 100)
        config.setdefault("n_workers", 1)
        config.setdefault("confidence_level", 0.95)

    if workflow == "profile_likelihood":
        parameter_names = list(config["parameters"])

        config.setdefault("profile_parameters", parameter_names)
        config.setdefault("profile_n_points", 15)
        config.setdefault("profile_span_factor", 10.0)
        config.setdefault("profile_log_space", True)

    return config


def project_state_to_gui_metadata(
    project_state: ProjectState,
) -> dict[str, Any]:
    """
    Build lightweight GUI metadata from ProjectState.
    """

    return {
        "project_name": project_state.project_name,
        "project_notes": project_state.project_notes,
        "schema_version": project_state.schema_version,
        "normalization_method": project_state.normalization_method,
        "last_fit_output_dir": project_state.last_fit_output_dir,
        "metadata": dict(project_state.metadata or {}),
    }


def project_state_to_gui_project_payload(
    project_state: ProjectState,
    *,
    workflow: str = "fit",
    use_variable_projection: bool = True,
    use_multispecies_variable_projection: bool = False,
    observed_species: str | list[str] | None = None,
) -> dict[str, Any]:
    """
    Build a GUI-friendly project payload from the existing ProjectState.

    This is not a replacement for project_state_to_dict().
    It is a convenience representation for GUI/API workflows.
    """

    return {
        "metadata": project_state_to_gui_metadata(project_state),
        "workflow": workflow,
        "config": project_state_to_backend_config(
            project_state,
            workflow=workflow,
            use_variable_projection=use_variable_projection,
            use_multispecies_variable_projection=(
                use_multispecies_variable_projection
            ),
            observed_species=observed_species,
        ),
    }
