from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from odefit.api.backend import fit_global_observables_from_config
from odefit.progress import ProgressCallback, ProgressTracker


@dataclass
class FCSModelFailure:
    model_name: str
    error_type: str
    error_message: str


def _as_model_text_mapping(config: dict[str, Any]) -> dict[str, str]:
    model_texts = config.get("model_texts")

    if model_texts is None:
        models = config.get("models")

        if isinstance(models, dict):
            model_texts = {
                str(name): str(value)
                for name, value in models.items()
            }
        elif isinstance(models, list):
            model_texts = {}

            for index, entry in enumerate(models):
                if isinstance(entry, dict):
                    name = str(entry.get("name", f"model_{index}"))
                    text = str(
                        entry.get("model_text")
                        or entry.get("text")
                        or entry.get("model")
                    )
                    model_texts[name] = text
                else:
                    model_texts[f"model_{index}"] = str(entry)

    if model_texts is None and "model_text" in config:
        model_texts = {
            str(config.get("model_name", "model")): str(config["model_text"])
        }

    if not isinstance(model_texts, dict) or not model_texts:
        raise ValueError(
            "FCS model comparison requires model_texts, models, or model_text."
        )

    return {
        str(name): str(text)
        for name, text in model_texts.items()
    }


def _get_model_specific_value(
    value,
    *,
    model_name: str,
    default=None,
):
    if isinstance(value, dict) and model_name in value:
        return value[model_name]

    if value is None:
        return default

    return value


def _build_single_model_config(
    base_config: dict[str, Any],
    *,
    model_name: str,
    model_text: str,
) -> dict[str, Any]:
    model_config = dict(base_config)

    model_config.pop("model_texts", None)
    model_config.pop("models", None)
    model_config.pop("parameters_by_model", None)
    model_config.pop("initial_conditions_by_model", None)
    model_config.pop("observed_species_by_model", None)
    model_config.pop("settings_by_model", None)

    model_config["model_name"] = model_name
    model_config["model_text"] = model_text
    model_config["use_variable_projection"] = bool(
        base_config.get("use_variable_projection", True)
    )

    parameters_by_model = base_config.get("parameters_by_model")
    initial_conditions_by_model = base_config.get("initial_conditions_by_model")
    observed_species_by_model = base_config.get("observed_species_by_model")
    settings_by_model = base_config.get("settings_by_model")

    parameters = _get_model_specific_value(
        parameters_by_model,
        model_name=model_name,
        default=base_config.get("parameters"),
    )

    initial_conditions = _get_model_specific_value(
        initial_conditions_by_model,
        model_name=model_name,
        default=base_config.get("initial_conditions"),
    )

    observed_species = _get_model_specific_value(
        observed_species_by_model,
        model_name=model_name,
        default=base_config.get("observed_species", "A"),
    )

    settings = _get_model_specific_value(
        settings_by_model,
        model_name=model_name,
        default=base_config.get("settings"),
    )

    if parameters is not None:
        model_config["parameters"] = parameters

    if initial_conditions is not None:
        model_config["initial_conditions"] = initial_conditions

    if observed_species is not None:
        model_config["observed_species"] = observed_species

    if settings is not None:
        model_config["settings"] = settings

    return model_config


def _statistic(result, name: str):
    statistics = getattr(result, "statistics", None)

    if isinstance(statistics, dict):
        return statistics.get(name)

    return getattr(result, name, None)


def _result_row(
    *,
    model_name: str,
    output: dict[str, Any],
) -> dict[str, Any]:
    result = output["result"]

    fitted_parameters = getattr(result, "fitted_parameters", {})

    row = {
        "model_name": model_name,
        "success": bool(getattr(result, "success", False)),
        "rss": _statistic(result, "rss"),
        "rmse": _statistic(result, "rmse"),
        "aic": _statistic(result, "aic"),
        "bic": _statistic(result, "bic"),
        "nfev": getattr(result, "nfev", None),
        "message": getattr(result, "message", ""),
    }

    if isinstance(fitted_parameters, dict):
        for name, value in fitted_parameters.items():
            row[f"parameter_{name}"] = value

    return row


def fit_fcs_model_from_config(
    config: dict[str, Any],
    *,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """
    Fit one FCS-style dataset/model using the existing global-observable backend.
    """

    tracker = ProgressTracker(
        stage="fcs_fit",
        total=1,
        callback=progress_callback,
        payload={
            "engine_name": config.get("engine_name", "reference"),
            "time_column": config.get("time_column", "tau"),
        },
    )

    tracker.emit(
        current=0,
        message="Starting FCS fit",
    )

    output = fit_global_observables_from_config(config)

    tracker.emit(
        current=1,
        message="Finished FCS fit",
        payload={
            "success": bool(getattr(output.get("result"), "success", False)),
        },
    )

    return output


def compare_fcs_models_from_config(
    config: dict[str, Any],
    *,
    progress_callback: ProgressCallback | None = None,
    sort_by: str | None = None,
    raise_on_failure: bool | None = None,
) -> dict[str, Any]:
    """
    Fit one FCS-style dataset against multiple candidate models.
    """

    model_texts = _as_model_text_mapping(config)
    model_items = list(model_texts.items())
    total = len(model_items)

    sort_column = sort_by or str(config.get("sort_by", "bic"))
    should_raise = bool(
        config.get("raise_on_failure", False)
        if raise_on_failure is None
        else raise_on_failure
    )

    tracker = ProgressTracker(
        stage="fcs_model_comparison",
        total=total,
        callback=progress_callback,
        payload={
            "engine_name": config.get("engine_name", "reference"),
            "time_column": config.get("time_column", "tau"),
            "sort_by": sort_column,
        },
    )

    tracker.emit(
        current=0,
        message=f"Starting FCS model comparison over {total} models",
    )

    rows: list[dict[str, Any]] = []
    fit_outputs: dict[str, dict[str, Any]] = {}
    failures: list[FCSModelFailure] = []

    for index, (model_name, model_text) in enumerate(model_items, start=1):
        tracker.emit(
            current=index - 1,
            message=f"Fitting model {model_name}",
            payload={
                "model_name": model_name,
            },
        )

        model_config = _build_single_model_config(
            config,
            model_name=model_name,
            model_text=model_text,
        )

        try:
            output = fit_global_observables_from_config(model_config)
            fit_outputs[model_name] = output

            row = _result_row(
                model_name=model_name,
                output=output,
            )
            rows.append(row)

            tracker.emit(
                current=index,
                message=f"Finished model {model_name}",
                payload={
                    "model_name": model_name,
                    "success": row["success"],
                    "rss": row.get("rss"),
                    "aic": row.get("aic"),
                    "bic": row.get("bic"),
                },
            )

        except Exception as exc:
            failure = FCSModelFailure(
                model_name=model_name,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            failures.append(failure)

            rows.append(
                {
                    "model_name": model_name,
                    "success": False,
                    "rss": None,
                    "rmse": None,
                    "aic": None,
                    "bic": None,
                    "nfev": None,
                    "message": str(exc),
                }
            )

            tracker.emit(
                current=index,
                message=f"Failed model {model_name}",
                payload={
                    "model_name": model_name,
                    "success": False,
                    "error_type": failure.error_type,
                    "error_message": failure.error_message,
                },
            )

            if should_raise:
                raise

    table = pd.DataFrame(rows)

    if sort_column in table.columns:
        table = table.sort_values(
            by=sort_column,
            na_position="last",
        ).reset_index(drop=True)

    if not table.empty:
        table.insert(0, "rank", range(1, len(table) + 1))

    tracker.emit(
        current=total,
        message="Finished FCS model comparison",
        payload={
            "n_models": total,
            "n_successful": int(table["success"].sum()) if "success" in table else 0,
            "n_failed": len(failures),
        },
    )

    return {
        "comparison_table": table,
        "fit_outputs": fit_outputs,
        "failures": [asdict(failure) for failure in failures],
        "sort_by": sort_column,
    }
