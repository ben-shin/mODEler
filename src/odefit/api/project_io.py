from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import uuid


PROJECT_SCHEMA_VERSION = "0.2"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [_json_safe(item) for item in value]

    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]

    return value


@dataclass
class ModelerProject:
    project_id: str
    schema_version: str
    created_at: str
    updated_at: str
    name: str
    description: str
    workflow: str
    config: dict[str, Any]
    ui_state: dict[str, Any] = field(default_factory=dict)
    result_payload: dict[str, Any] | None = None
    notes: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "name": self.name,
            "description": self.description,
            "workflow": self.workflow,
            "config": _json_safe(self.config),
            "ui_state": _json_safe(self.ui_state),
            "result_payload": _json_safe(self.result_payload),
            "notes": list(self.notes),
            "tags": list(self.tags),
        }


def create_project(
    *,
    name: str,
    workflow: str,
    config: dict[str, Any],
    description: str = "",
    ui_state: dict[str, Any] | None = None,
    result_payload: dict[str, Any] | None = None,
    notes: list[str] | None = None,
    tags: list[str] | None = None,
    project_id: str | None = None,
) -> ModelerProject:
    now = _utc_now_iso()

    return ModelerProject(
        project_id=project_id or str(uuid.uuid4()),
        schema_version=PROJECT_SCHEMA_VERSION,
        created_at=now,
        updated_at=now,
        name=name,
        description=description,
        workflow=workflow,
        config=config,
        ui_state=ui_state or {},
        result_payload=result_payload,
        notes=notes or [],
        tags=tags or [],
    )


def validate_project_dict(project: dict[str, Any]) -> None:
    required_keys = {
        "project_id",
        "schema_version",
        "created_at",
        "updated_at",
        "name",
        "workflow",
        "config",
    }

    missing = required_keys - set(project)

    if missing:
        raise ValueError(
            "Project file is missing required keys: "
            + ", ".join(sorted(missing))
        )

    if str(project["schema_version"]) != PROJECT_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported project schema version: "
            f"{project['schema_version']}. "
            f"Expected {PROJECT_SCHEMA_VERSION}."
        )

    if not isinstance(project["config"], dict):
        raise ValueError("Project 'config' must be a dictionary.")

    if not isinstance(project.get("ui_state", {}), dict):
        raise ValueError("Project 'ui_state' must be a dictionary.")

    if project.get("result_payload") is not None and not isinstance(
        project["result_payload"],
        dict,
    ):
        raise ValueError("Project 'result_payload' must be a dictionary or null.")

    if not isinstance(project.get("notes", []), list):
        raise ValueError("Project 'notes' must be a list.")

    if not isinstance(project.get("tags", []), list):
        raise ValueError("Project 'tags' must be a list.")


def project_from_dict(project: dict[str, Any]) -> ModelerProject:
    validate_project_dict(project)

    return ModelerProject(
        project_id=str(project["project_id"]),
        schema_version=str(project["schema_version"]),
        created_at=str(project["created_at"]),
        updated_at=str(project["updated_at"]),
        name=str(project["name"]),
        description=str(project.get("description", "")),
        workflow=str(project["workflow"]),
        config=dict(project["config"]),
        ui_state=dict(project.get("ui_state", {})),
        result_payload=project.get("result_payload"),
        notes=list(project.get("notes", [])),
        tags=list(project.get("tags", [])),
    )


def save_project(
    project: ModelerProject,
    file_path: str | Path,
) -> Path:
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    project.updated_at = _utc_now_iso()

    with path.open("w") as handle:
        json.dump(
            project.to_dict(),
            handle,
            indent=2,
        )

    return path


def load_project(file_path: str | Path) -> ModelerProject:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Project file does not exist: {path}")

    with path.open("r") as handle:
        data = json.load(handle)

    return project_from_dict(data)


def update_project_config(
    project: ModelerProject,
    config_updates: dict[str, Any],
) -> ModelerProject:
    project.config.update(config_updates)
    project.updated_at = _utc_now_iso()

    return project


def attach_result_payload(
    project: ModelerProject,
    result_payload: dict[str, Any],
) -> ModelerProject:
    project.result_payload = result_payload
    project.updated_at = _utc_now_iso()

    return project


def project_to_config(project: ModelerProject) -> dict[str, Any]:
    return dict(project.config)


def summarize_project(project: ModelerProject) -> dict[str, Any]:
    return {
        "project_id": project.project_id,
        "schema_version": project.schema_version,
        "name": project.name,
        "description": project.description,
        "workflow": project.workflow,
        "created_at": project.created_at,
        "updated_at": project.updated_at,
        "has_result_payload": project.result_payload is not None,
        "n_notes": len(project.notes),
        "tags": list(project.tags),
    }
