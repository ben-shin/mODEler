from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from odefit.api.backend import validate_backend_engine_name


DEFAULT_PROJECT_SCHEMA_VERSION = 1
DEFAULT_ENGINE_NAME = "reference"


class ProjectConfigError(ValueError):
    pass


def _deepcopy_mapping(payload: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(payload)


def looks_like_fit_config(payload: dict[str, Any]) -> bool:
    """
    Heuristic for identifying raw fit/FCS/model-comparison configs.

    This intentionally avoids being too strict because configs evolve quickly.
    """

    fit_keys = {
        "engine_name",
        "data",
        "time_column",
        "signal_columns",
        "model_text",
        "model_texts",
        "models",
        "parameters",
        "parameters_by_model",
        "initial_conditions",
        "initial_conditions_by_model",
        "observed_species",
        "observed_species_by_model",
        "use_variable_projection",
    }

    return any(key in payload for key in fit_keys)


def ensure_engine_name(
    config: dict[str, Any],
    *,
    default_engine_name: str = DEFAULT_ENGINE_NAME,
) -> dict[str, Any]:
    """
    Return a copy of one config with engine_name guaranteed to exist.

    Existing engine_name values are preserved exactly.
    """

    output = _deepcopy_mapping(config)

    if not output.get("engine_name"):
        output["engine_name"] = default_engine_name

    return output


def validate_project_engine_name(config: dict[str, Any]) -> dict[str, Any]:
    """
    Validate the engine_name in a config and return a GUI-friendly payload.

    This does not mutate the config.
    """

    engine_name = str(config.get("engine_name", DEFAULT_ENGINE_NAME))

    return validate_backend_engine_name(engine_name)


def _normalize_known_config_container(
    payload: dict[str, Any],
    *,
    key: str,
    default_engine_name: str,
) -> None:
    value = payload.get(key)

    if isinstance(value, dict):
        payload[key] = ensure_engine_name(
            value,
            default_engine_name=default_engine_name,
        )


def _normalize_workflow_configs(
    payload: dict[str, Any],
    *,
    default_engine_name: str,
) -> None:
    workflow_configs = payload.get("workflow_configs")

    if isinstance(workflow_configs, dict):
        payload["workflow_configs"] = {
            name: (
                ensure_engine_name(
                    config,
                    default_engine_name=default_engine_name,
                )
                if isinstance(config, dict)
                else config
            )
            for name, config in workflow_configs.items()
        }

    elif isinstance(workflow_configs, list):
        normalized = []

        for item in workflow_configs:
            if isinstance(item, dict):
                normalized.append(
                    ensure_engine_name(
                        item,
                        default_engine_name=default_engine_name,
                    )
                )
            else:
                normalized.append(item)

        payload["workflow_configs"] = normalized


def normalize_project_payload(
    payload: dict[str, Any],
    *,
    default_engine_name: str = DEFAULT_ENGINE_NAME,
) -> dict[str, Any]:
    """
    Normalize a project/config payload for save/load.

    Supported shapes:

    1. Raw fit config:
       {
         "engine_name": "numba_projection",
         "data": "...",
         ...
       }

    2. Project payload:
       {
         "project_name": "...",
         "fit_config": {...}
       }

    3. GUI-style project payload:
       {
         "project": {...},
         "workflow_configs": {
             "main_fit": {...},
             "bootstrap": {...}
         }
       }

    Unknown keys are preserved.
    """

    if not isinstance(payload, dict):
        raise ProjectConfigError("Project payload must be a dictionary.")

    output = _deepcopy_mapping(payload)

    output.setdefault(
        "project_schema_version",
        DEFAULT_PROJECT_SCHEMA_VERSION,
    )

    # Raw config case.
    if looks_like_fit_config(output):
        output = ensure_engine_name(
            output,
            default_engine_name=default_engine_name,
        )

    # Nested config cases.
    for key in [
        "fit_config",
        "config",
        "model_config",
        "comparison_config",
        "fcs_config",
        "surface_config",
    ]:
        _normalize_known_config_container(
            output,
            key=key,
            default_engine_name=default_engine_name,
        )

    _normalize_workflow_configs(
        output,
        default_engine_name=default_engine_name,
    )

    return output


def save_project_payload(
    payload: dict[str, Any],
    path: str | Path,
    *,
    default_engine_name: str = DEFAULT_ENGINE_NAME,
    indent: int = 2,
) -> Path:
    """
    Save a normalized project/config payload to JSON.
    """

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    normalized = normalize_project_payload(
        payload,
        default_engine_name=default_engine_name,
    )

    with output_path.open("w") as handle:
        json.dump(
            normalized,
            handle,
            indent=indent,
            sort_keys=True,
        )

    return output_path


def load_project_payload(
    path: str | Path,
    *,
    default_engine_name: str = DEFAULT_ENGINE_NAME,
) -> dict[str, Any]:
    """
    Load a project/config payload from JSON and normalize engine_name fields.
    """

    input_path = Path(path)

    with input_path.open() as handle:
        payload = json.load(handle)

    return normalize_project_payload(
        payload,
        default_engine_name=default_engine_name,
    )


def collect_engine_names(payload: dict[str, Any]) -> dict[str, str]:
    """
    Collect engine_name values from known project/config locations.

    Returns a path-like mapping useful for GUI/debug display.
    """

    normalized = normalize_project_payload(payload)

    engine_names: dict[str, str] = {}

    if looks_like_fit_config(normalized):
        engine_names["/"] = str(normalized.get("engine_name", DEFAULT_ENGINE_NAME))

    for key in [
        "fit_config",
        "config",
        "model_config",
        "comparison_config",
        "fcs_config",
        "surface_config",
    ]:
        value = normalized.get(key)

        if isinstance(value, dict):
            engine_names[f"/{key}"] = str(
                value.get("engine_name", DEFAULT_ENGINE_NAME)
            )

    workflow_configs = normalized.get("workflow_configs")

    if isinstance(workflow_configs, dict):
        for name, config in workflow_configs.items():
            if isinstance(config, dict):
                engine_names[f"/workflow_configs/{name}"] = str(
                    config.get("engine_name", DEFAULT_ENGINE_NAME)
                )

    elif isinstance(workflow_configs, list):
        for index, config in enumerate(workflow_configs):
            if isinstance(config, dict):
                engine_names[f"/workflow_configs/{index}"] = str(
                    config.get("engine_name", DEFAULT_ENGINE_NAME)
                )

    return engine_names


def validate_project_engines(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Validate all discovered engine_name values in a project payload.
    """

    engine_names = collect_engine_names(payload)

    validations = {
        path: validate_backend_engine_name(engine_name)
        for path, engine_name in engine_names.items()
    }

    all_valid = all(
        validation.get("valid", False)
        for validation in validations.values()
    )

    return {
        "valid": all_valid,
        "engine_names": engine_names,
        "validations": validations,
    }
