from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from odefit.api.fcs import compare_fcs_models_from_config
from odefit.progress import ProgressEvent


ARROW_PATTERN = re.compile(r"<->|->|<-")
PARAMETER_PATTERN = re.compile(r"\b[kK][A-Za-z0-9_]*\b")
SPECIES_PATTERN = re.compile(r"\b[A-Za-z][A-Za-z0-9_]*\b")


IGNORE_SPECIES_TOKENS = {
    "k",
    "kf",
    "kr",
    "time",
    "tau",
    "t",
}


def on_progress(event: ProgressEvent) -> None:
    print(event.to_console_line(), flush=True)


def read_model_files(model_dir: Path) -> dict[str, str]:
    model_texts: dict[str, str] = {}

    for path in sorted(model_dir.glob("*.txt")):
        text = path.read_text().strip()

        if not text:
            continue

        model_texts[path.stem] = text

    if not model_texts:
        raise ValueError(f"No non-empty .txt model files found in {model_dir}")

    return model_texts


def infer_time_and_signal_columns(
    dataframe: pd.DataFrame,
    *,
    time_column: str | None,
    signal_columns: list[str] | None,
) -> tuple[str, list[str]]:
    numeric_columns = dataframe.select_dtypes(include="number").columns.tolist()

    if not numeric_columns:
        raise ValueError("The FCS CSV has no numeric columns.")

    if time_column is None:
        # Prefer common FCS/time names if present.
        candidates = [
            "tau",
            "lag_time",
            "lag",
            "time",
            "t",
            "Time",
            "Tau",
        ]

        for candidate in candidates:
            if candidate in dataframe.columns:
                time_column = candidate
                break

    if time_column is None:
        time_column = numeric_columns[0]

    if time_column not in dataframe.columns:
        raise ValueError(
            f"time_column={time_column!r} is not in the CSV columns: "
            f"{list(dataframe.columns)}"
        )

    if signal_columns is None:
        signal_columns = [
            column
            for column in numeric_columns
            if column != time_column
        ]

    if not signal_columns:
        raise ValueError("No signal columns selected.")

    missing = [
        column
        for column in signal_columns
        if column not in dataframe.columns
    ]

    if missing:
        raise ValueError(f"Signal columns not found in CSV: {missing}")

    return time_column, signal_columns


def infer_parameter_names(model_text: str) -> list[str]:
    """
    Try to infer parameter names.

    If the model text explicitly contains k-like tokens, use those.
    Otherwise generate default names based on reaction arrows:
        A -> B     gives k1f
        A <-> B    gives k1f and k1r
    """

    explicit = sorted(set(PARAMETER_PATTERN.findall(model_text)))

    if explicit:
        return explicit

    parameter_names: list[str] = []

    arrows = ARROW_PATTERN.findall(model_text)

    for index, arrow in enumerate(arrows, start=1):
        parameter_names.append(f"k{index}f")

        if arrow == "<->":
            parameter_names.append(f"k{index}r")

    if not parameter_names:
        parameter_names = ["k1f"]

    return parameter_names


def infer_species_names(model_text: str) -> list[str]:
    species: list[str] = []

    for line in model_text.splitlines():
        line = line.strip()

        if not line or line.startswith("#"):
            continue

        # Remove obvious parameter-like tokens first.
        line_without_parameters = PARAMETER_PATTERN.sub(" ", line)

        for token in SPECIES_PATTERN.findall(line_without_parameters):
            if token in IGNORE_SPECIES_TOKENS:
                continue

            if token.startswith("k") or token.startswith("K"):
                continue

            if token not in species:
                species.append(token)

    if not species:
        species = ["A", "B"]

    return species


def build_parameters_by_model(
    model_texts: dict[str, str],
    *,
    initial_guess: float,
    lower_bound: float,
    upper_bound: float,
) -> dict[str, dict[str, dict[str, float]]]:
    parameters_by_model = {}

    for model_name, model_text in model_texts.items():
        parameter_names = infer_parameter_names(model_text)

        parameters_by_model[model_name] = {
            parameter_name: {
                "initial_guess": initial_guess,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
            }
            for parameter_name in parameter_names
        }

    return parameters_by_model


def build_initial_conditions_by_model(
    model_texts: dict[str, str],
    *,
    initial_active_species: str | None,
) -> dict[str, dict[str, dict[str, Any]]]:
    initial_conditions_by_model = {}

    for model_name, model_text in model_texts.items():
        species_names = infer_species_names(model_text)

        active_species = initial_active_species or species_names[0]

        initial_conditions = {}

        for species in species_names:
            initial_conditions[species] = {
                "value": 1.0 if species == active_species else 0.0,
                "mode": "fixed",
            }

        initial_conditions_by_model[model_name] = initial_conditions

    return initial_conditions_by_model


def write_model_preview(
    *,
    output_dir: Path,
    model_texts: dict[str, str],
    parameters_by_model: dict[str, dict[str, dict[str, float]]],
    initial_conditions_by_model: dict[str, dict[str, dict[str, Any]]],
) -> None:
    preview_rows = []

    for model_name, model_text in model_texts.items():
        preview_rows.append(
            {
                "model_name": model_name,
                "model_text": model_text,
                "parameters": ", ".join(parameters_by_model[model_name]),
                "species": ", ".join(initial_conditions_by_model[model_name]),
            }
        )

    pd.DataFrame(preview_rows).to_csv(
        output_dir / "model_preview.csv",
        index=False,
    )


def json_default(value):
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")

    if hasattr(value, "to_dict"):
        try:
            return value.to_dict()
        except Exception:
            pass

    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fit one FCS CSV against every .txt ODE model in a directory "
            "using the FCS API progress wrapper."
        )
    )

    parser.add_argument(
        "--data",
        required=True,
        help="Path to FCS CSV.",
    )

    parser.add_argument(
        "--model-dir",
        required=True,
        help="Directory containing .txt model files.",
    )

    parser.add_argument(
        "--output-dir",
        default="fcs_ode/fits_all_txt_models_api",
    )

    parser.add_argument(
        "--engine-name",
        default="reference",
        help="Backend engine name: reference, numba_projection, jax_projection.",
    )

    parser.add_argument(
        "--time-column",
        default=None,
        help="Time/lag-time column. If omitted, inferred from CSV.",
    )

    parser.add_argument(
        "--signal-columns",
        nargs="+",
        default=None,
        help="Signal columns. If omitted, all numeric columns except time column.",
    )

    parser.add_argument(
        "--observed-species",
        default=None,
        help=(
            "Observed species for projection. If omitted, uses the first species "
            "in each model via observed_species_by_model."
        ),
    )

    parser.add_argument(
        "--initial-active-species",
        default=None,
        help=(
            "Species initialized to 1.0. If omitted, first inferred species "
            "in each model is used."
        ),
    )

    parser.add_argument(
        "--allow-raw-fcs-matrix",
        action="store_true",
        help=(
            "Allow fitting many raw G(tau) columns directly over elapsed time. "
            "Usually not recommended unless this is intentional."
        ),
    )

    parser.add_argument("--initial-guess", type=float, default=0.1)
    parser.add_argument("--lower-bound", type=float, default=1e-8)
    parser.add_argument("--upper-bound", type=float, default=100.0)
    parser.add_argument("--max-nfev", type=int, default=300)
    parser.add_argument("--rtol", type=float, default=1e-7)
    parser.add_argument("--atol", type=float, default=1e-10)
    parser.add_argument("--sort-by", default="bic")
    parser.add_argument("--no-progress", action="store_true")

    args = parser.parse_args()

    data_path = Path(args.data)
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    dataframe = pd.read_csv(data_path)

    time_column, signal_columns = infer_time_and_signal_columns(
        dataframe,
        time_column=args.time_column,
        signal_columns=args.signal_columns,
    )

    model_texts = read_model_files(model_dir)

    parameters_by_model = build_parameters_by_model(
        model_texts,
        initial_guess=args.initial_guess,
        lower_bound=args.lower_bound,
        upper_bound=args.upper_bound,
    )

    initial_conditions_by_model = build_initial_conditions_by_model(
        model_texts,
        initial_active_species=args.initial_active_species,
    )

    observed_species_by_model = {}

    for model_name, initial_conditions in initial_conditions_by_model.items():
        if args.observed_species is not None:
            observed_species_by_model[model_name] = args.observed_species
        else:
            observed_species_by_model[model_name] = next(iter(initial_conditions))

    config = {
        "engine_name": args.engine_name,
        "data": str(data_path),
        "time_column": time_column,
        "signal_columns": signal_columns,
        "observed_species_by_model": observed_species_by_model,
        "use_variable_projection": True,
        "fit_scale": True,
        "fit_offset": True,
        "model_texts": model_texts,
        "parameters_by_model": parameters_by_model,
        "initial_conditions_by_model": initial_conditions_by_model,
        "method": "trf",
        "loss": "linear",
        "max_nfev": args.max_nfev,
        "rtol": args.rtol,
        "atol": args.atol,
        "variable_projection_backend": "numpy",
        "variable_projection_method": "LSODA",
        "show_progress": False,
        "no_plots": True,
        "sort_by": args.sort_by,
    }

    generated_config_path = output_dir / "generated_fcs_all_txt_config.json"

    with generated_config_path.open("w") as handle:
        json.dump(config, handle, indent=2)

    write_model_preview(
        output_dir=output_dir,
        model_texts=model_texts,
        parameters_by_model=parameters_by_model,
        initial_conditions_by_model=initial_conditions_by_model,
    )

    print("\nFCS all-model run")
    print("=================")
    print(f"Data: {data_path}")
    print(f"Model dir: {model_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Engine: {args.engine_name}")
    print(f"Time column: {time_column}")
    print(f"Signal columns: {len(signal_columns)}")
    
    if len(signal_columns) > 100 and "time" in time_column.lower():
        print("\nWARNING")
        print("=======")
        print(
            "You selected many signal columns with an elapsed-time column. "
            "If this is a raw FCS matrix, this probably means you are fitting "
            "G(tau) columns as kinetic traces over elapsed time."
        )
        print(
            "For raw FCS data, first extract FCS features per elapsed timepoint, "
            "then fit those feature timecourses to ODE models."
        )
        print(
            "Rerun with --allow-raw-fcs-matrix if you intentionally want this."
        )

        if not getattr(args, "allow_raw_fcs_matrix", False):
            raise SystemExit(2)

    print(f"Models: {len(model_texts)}")
    print(f"Generated config: {generated_config_path}")

    output = compare_fcs_models_from_config(
        config,
        progress_callback=None if args.no_progress else on_progress,
        sort_by=args.sort_by,
    )

    comparison_table = output["comparison_table"]

    table_path = output_dir / "fcs_all_txt_model_comparison_table.csv"
    summary_path = output_dir / "fcs_all_txt_model_comparison_summary.json"
    failures_path = output_dir / "fcs_all_txt_model_comparison_failures.json"

    comparison_table.to_csv(table_path, index=False)

    with failures_path.open("w") as handle:
        json.dump(output["failures"], handle, indent=2)

    summary_payload = {
        "sort_by": output["sort_by"],
        "data": str(data_path),
        "model_dir": str(model_dir),
        "engine_name": args.engine_name,
        "time_column": time_column,
        "signal_columns": signal_columns,
        "comparison_table": comparison_table.to_dict(orient="records"),
        "failures": output["failures"],
        "model_names": list(model_texts),
    }

    with summary_path.open("w") as handle:
        json.dump(summary_payload, handle, indent=2, default=json_default)
    
    export_fcs_fit_outputs(
        output_dir=output_dir,
        fit_outputs=output["fit_outputs"],
    )

    print("\nFCS all-model comparison complete")
    print("=================================")
    print(comparison_table.to_string(index=False))

    print("\nWritten files:")
    print(f"  table: {table_path}")
    print(f"  summary: {summary_path}")
    print(f"  failures: {failures_path}")
    print(f"  model preview: {output_dir / 'model_preview.csv'}")
    print(f"  generated config: {generated_config_path}")
    print(f"  per-model fits: {output_dir / 'models'}")
    print(f"  per-model summary: {output_dir / 'per_model_fit_summary.csv'}")

def _safe_model_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_")


def _result_statistics(result) -> dict[str, Any]:
    statistics = getattr(result, "statistics", None)

    if isinstance(statistics, dict):
        return dict(statistics)

    output = {}

    for key in ["rss", "rmse", "aic", "bic"]:
        if hasattr(result, key):
            output[key] = getattr(result, key)

    return output


def _result_parameters(result) -> dict[str, Any]:
    parameters = getattr(result, "fitted_parameters", None)

    if isinstance(parameters, dict):
        return dict(parameters)

    return {}


def export_fcs_fit_outputs(
    *,
    output_dir: Path,
    fit_outputs: dict[str, dict[str, Any]],
) -> None:
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for model_name, fit_output in fit_outputs.items():
        result = fit_output.get("result")

        if result is None:
            continue

        safe_name = _safe_model_name(model_name)
        model_dir = models_dir / safe_name
        model_dir.mkdir(parents=True, exist_ok=True)

        observable_table = getattr(result, "observable_table", None)
        predicted_dataframe = getattr(result, "predicted_dataframe", None)
        residuals_dataframe = getattr(result, "residuals_dataframe", None)

        if isinstance(observable_table, pd.DataFrame):
            observable_table.to_csv(model_dir / "observable_table.csv", index=False)

        if isinstance(predicted_dataframe, pd.DataFrame):
            predicted_dataframe.to_csv(model_dir / "predicted.csv", index=False)

        if isinstance(residuals_dataframe, pd.DataFrame):
            residuals_dataframe.to_csv(model_dir / "residuals.csv", index=False)

        statistics = _result_statistics(result)
        parameters = _result_parameters(result)

        fit_summary = {
            "model_name": model_name,
            "success": bool(getattr(result, "success", False)),
            "message": str(getattr(result, "message", "")),
            "nfev": getattr(result, "nfev", None),
            "statistics": statistics,
            "fitted_parameters": parameters,
        }

        with (model_dir / "fit_summary.json").open("w") as handle:
            json.dump(fit_summary, handle, indent=2, default=json_default)

        row = {
            "model_name": model_name,
            "model_dir": str(model_dir),
            "success": fit_summary["success"],
            "message": fit_summary["message"],
            "nfev": fit_summary["nfev"],
        }

        for key, value in statistics.items():
            row[key] = value

        for key, value in parameters.items():
            row[f"parameter_{key}"] = value

        summary_rows.append(row)

    pd.DataFrame(summary_rows).to_csv(
        output_dir / "per_model_fit_summary.csv",
        index=False,
        )

if __name__ == "__main__":
    main()
