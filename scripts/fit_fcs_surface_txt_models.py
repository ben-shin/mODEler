from __future__ import annotations

import argparse
import json
import math
import re
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from odefit.engines.registry import get_engine_bundle
from odefit.fitting.engine_helpers import engine_solve_to_dataframe
from odefit.fitting.fit_settings import FitSettings
from odefit.model.model_spec import build_model_spec
from odefit.progress import ProgressTracker


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


@dataclass
class FCSSurfaceFitResult:
    model_name: str
    success: bool
    message: str
    nfev: int
    rss: float
    rmse: float
    aic: float
    bic: float
    n_observations: int
    n_parameters: int
    baseline: float
    scale: float
    tau_d_ms: float
    alpha: float
    fitted_parameters: dict[str, float]
    output_dir: str


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_")

def _name_from_model_item(item) -> str:
    """
    Extract a stable string name from ModelSpec species/parameter entries.

    Handles:
      - plain strings: "P1"
      - objects with .name
      - dicts with "name"
      - repr strings like "Species(name='P1')"
      - repr strings like "Parameter(name='k1f')"
    """

    if isinstance(item, str):
        text = item
    else:
        name = getattr(item, "name", None)

        if name is not None:
            return str(name)

        if isinstance(item, dict) and "name" in item:
            return str(item["name"])

        text = str(item)

    # Handle repr-like text: Species(name='P1'), Parameter(name="k1f")
    match = re.search(r"name=['\"]([^'\"]+)['\"]", text)

    if match:
        return match.group(1)

    # Handle simple dataclass-ish text: Species(P1)
    match = re.search(r"\(([A-Za-z][A-Za-z0-9_]*)\)", text)

    if match:
        return match.group(1)

    return text

def model_parameter_names(model) -> list[str]:
    parameters = getattr(model, "parameters", [])

    if isinstance(parameters, dict):
        return [str(name) for name in parameters.keys()]

    return [_name_from_model_item(item) for item in parameters]


def model_species_names(model) -> list[str]:
    species = getattr(model, "species", [])

    if isinstance(species, dict):
        return [str(name) for name in species.keys()]

    return [_name_from_model_item(item) for item in species]

def json_default(value):
    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")

    return str(value)


def parse_tau_from_column(column: str, fallback_index: int) -> float:
    text = str(column)

    if text.startswith("G_tau_"):
        text = text.removeprefix("G_tau_")

    if text.endswith("_ms"):
        text = text.removesuffix("_ms")

    text = text.replace("p", ".")

    try:
        return float(text)
    except ValueError:
        pass

    matches = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)

    if matches:
        return float(matches[-1])

    return float(fallback_index)


def load_raw_fcs_matrix(
    *,
    data_path: Path,
    time_column: str,
) -> tuple[np.ndarray, list[str], np.ndarray, pd.DataFrame, np.ndarray]:
    dataframe = pd.read_csv(data_path)

    if time_column not in dataframe.columns:
        raise ValueError(
            f"time_column={time_column!r} not found. "
            f"Available columns include {list(dataframe.columns[:10])}"
        )

    numeric_columns = dataframe.select_dtypes(include="number").columns.tolist()

    signal_columns = [
        column
        for column in numeric_columns
        if str(column).startswith("G_tau_")
    ]

    if not signal_columns:
        signal_columns = [
            column
            for column in numeric_columns
            if column != time_column
        ]

    if not signal_columns:
        raise ValueError("No numeric FCS signal columns found.")

    tau_values = np.asarray(
        [
            parse_tau_from_column(column, index)
            for index, column in enumerate(signal_columns)
        ],
        dtype=float,
    )

    order = np.argsort(tau_values)

    tau_values = tau_values[order]
    signal_columns = [signal_columns[index] for index in order]

    time_values = dataframe[time_column].to_numpy(dtype=float)
    observed_matrix = dataframe[signal_columns].to_numpy(dtype=float)

    return time_values, signal_columns, tau_values, dataframe, observed_matrix


def read_model_files(model_dir: Path) -> dict[str, str]:
    model_texts: dict[str, str] = {}

    for path in sorted(model_dir.glob("*.txt")):
        text = path.read_text().strip()

        if text:
            model_texts[path.stem] = text

    if not model_texts:
        raise ValueError(f"No non-empty .txt model files found in {model_dir}")

    return model_texts


def infer_parameter_names(model_text: str) -> list[str]:
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


def build_initial_conditions(
    species_names: list[str],
    *,
    active_species: str | None,
) -> dict[str, float]:
    if not species_names:
        raise ValueError("No species names available.")

    species_names = [str(species) for species in species_names]
    active = str(active_species) if active_species is not None else species_names[0]

    return {
        species: 1.0 if species == active else 0.0
        for species in species_names
    }

def complete_initial_conditions_for_model(
    *,
    model,
    initial_conditions: dict[str, float],
    active_species: str | None,
) -> dict[str, float]:
    """
    Make sure every species required by the parsed ModelSpec has an initial condition.
    """

    species_names = model_species_names(model)

    completed = {
        str(key): float(value)
        for key, value in initial_conditions.items()
    }

    if active_species is not None:
        active = str(active_species)
    elif species_names:
        active = str(species_names[0])
    else:
        active = None

    for species in species_names:
        species = str(species)

        if species not in completed:
            completed[species] = 1.0 if species == active else 0.0

    return completed

def anomalous_fcs_kernel(
    tau_ms: np.ndarray,
    *,
    tau_d_ms: float,
    alpha: float,
) -> np.ndarray:
    tau_d_ms = max(float(tau_d_ms), 1e-30)
    alpha = max(float(alpha), 1e-12)

    return 1.0 / np.power(1.0 + tau_ms / tau_d_ms, alpha)


def solve_linear_baseline_scale(
    *,
    basis_matrix: np.ndarray,
    observed_matrix: np.ndarray,
) -> tuple[float, float, np.ndarray, np.ndarray, float]:
    q = np.asarray(basis_matrix, dtype=float).reshape(-1)
    y = np.asarray(observed_matrix, dtype=float).reshape(-1)

    valid = np.isfinite(q) & np.isfinite(y)

    if valid.sum() < 2:
        raise ValueError("Not enough finite observations to solve baseline/scale.")

    design = np.column_stack(
        [
            np.ones(valid.sum(), dtype=float),
            q[valid],
        ]
    )

    beta, *_ = np.linalg.lstsq(
        design,
        y[valid],
        rcond=None,
    )

    baseline = float(beta[0])
    scale = float(beta[1])

    predicted = baseline + scale * basis_matrix
    residuals = observed_matrix - predicted

    rss = float(np.nansum(residuals[valid.reshape(observed_matrix.shape)] ** 2))

    return baseline, scale, predicted, residuals, rss


def calculate_statistics(
    *,
    rss: float,
    n_observations: int,
    n_parameters: int,
) -> tuple[float, float, float]:
    n = max(int(n_observations), 1)
    k = max(int(n_parameters), 1)

    rmse = math.sqrt(max(rss, 0.0) / n)

    if rss <= 0:
        aic = -math.inf
        bic = -math.inf
    else:
        aic = n * math.log(rss / n) + 2 * k
        bic = n * math.log(rss / n) + k * math.log(n)

    return rmse, aic, bic


def make_large_residual(observed_matrix: np.ndarray, value: float = 1e6) -> np.ndarray:
    return np.full(
        int(np.isfinite(observed_matrix).sum()),
        float(value),
        dtype=float,
    )


def simulate_species(
    *,
    engine_bundle,
    model,
    parameters: dict[str, float],
    initial_conditions: dict[str, float],
    time_values: np.ndarray,
    observed_species: str,
    settings: FitSettings,
) -> np.ndarray:
    simulation = engine_solve_to_dataframe(
        engine_bundle=engine_bundle,
        model=model,
        parameters=parameters,
        initial_conditions=initial_conditions,
        timepoints=time_values,
        settings=settings.to_simulation_settings()
        if hasattr(settings, "to_simulation_settings")
        else None,
    )

    if observed_species not in simulation.columns:
        raise ValueError(
            f"Observed species {observed_species!r} not found in simulation columns "
            f"{list(simulation.columns)}"
        )

    values = simulation[observed_species].to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError("Simulation produced non-finite observed species values.")

    return values


def fit_one_surface_model(
    *,
    model_name: str,
    model_text: str,
    time_values: np.ndarray,
    tau_values: np.ndarray,
    observed_matrix: np.ndarray,
    output_dir: Path,
    engine_name: str,
    observed_species: str | None,
    active_species: str | None,
    initial_guess: float,
    lower_bound: float,
    upper_bound: float,
    tau_d_initial_ms: float | None,
    tau_d_lower_ms: float,
    tau_d_upper_ms: float,
    alpha_initial: float,
    alpha_lower: float,
    alpha_upper: float,
    max_nfev: int,
    rtol: float,
    atol: float,
    heartbeat_seconds: float,
) -> FCSSurfaceFitResult:
    model = build_model_spec(
            model_text,
            name=model_name,
    )

    parameter_names = model_parameter_names(model)
    species_names = model_species_names(model)

    if not parameter_names:
        parameter_names = infer_parameter_names(model_text)

    if not species_names:
        parameter_names = infer_species_names(model_text)

    parameter_names = [str(name) for name in parameter_names]
    species_names = [str(name) for name in species_names]

    selected_species = observed_species or species_names[0]

    initial_conditions = build_initial_conditions(
        species_names,
        active_species=active_species,
    )

    initial_conditions = complete_initial_conditions_for_model(
        model=model,
        initial_conditions=initial_conditions,
        active_species=active_species,
    )

    engine_bundle = get_engine_bundle(engine_name)

    settings = FitSettings(
        species_mapping={},
        use_normalized_data=False,
        method="trf",
        loss="linear",
        max_nfev=max_nfev,
        rtol=rtol,
        atol=atol,
    )

    if tau_d_initial_ms is None:
        positive_tau = tau_values[tau_values > 0]
        tau_d_initial_ms = float(np.median(positive_tau)) if len(positive_tau) else 1.0

    lower_log_rates = np.log10(np.full(len(parameter_names), lower_bound))
    upper_log_rates = np.log10(np.full(len(parameter_names), upper_bound))
    initial_log_rates = np.log10(np.full(len(parameter_names), initial_guess))

    x0 = np.concatenate(
        [
            initial_log_rates,
            [math.log10(tau_d_initial_ms), math.log10(alpha_initial)],
        ]
    )

    lower = np.concatenate(
        [
            lower_log_rates,
            [math.log10(tau_d_lower_ms), math.log10(alpha_lower)],
        ]
    )

    upper = np.concatenate(
        [
            upper_log_rates,
            [math.log10(tau_d_upper_ms), math.log10(alpha_upper)],
        ]
    )

    finite_mask = np.isfinite(observed_matrix)

    state = {
        "nfev": 0,
        "last_heartbeat": time.perf_counter(),
        "best_rss": math.inf,
        "best_payload": None,
        "last_error_type": None,
        "last_error_message": None,
        "last_traceback": None,
    }

    start_time = time.perf_counter()

    def unpack(theta: np.ndarray) -> tuple[dict[str, float], float, float]:
        rates = {
            name: float(10.0 ** theta[index])
            for index, name in enumerate(parameter_names)
        }

        tau_d_ms = float(10.0 ** theta[len(parameter_names)])
        alpha = float(10.0 ** theta[len(parameter_names) + 1])

        return rates, tau_d_ms, alpha

    def residual_function(theta: np.ndarray) -> np.ndarray:
        state["nfev"] += 1

        try:
            rates, tau_d_ms, alpha = unpack(theta)

            species_values = simulate_species(
                engine_bundle=engine_bundle,
                model=model,
                parameters=rates,
                initial_conditions=initial_conditions,
                time_values=time_values,
                observed_species=selected_species,
                settings=settings,
            )

            fcs_shape = anomalous_fcs_kernel(
                tau_values,
                tau_d_ms=tau_d_ms,
                alpha=alpha,
            )

            basis = species_values[:, None] * fcs_shape[None, :]

            baseline, scale, predicted, residuals, rss = solve_linear_baseline_scale(
                basis_matrix=basis,
                observed_matrix=observed_matrix,
            )

            if rss < state["best_rss"]:
                state["best_rss"] = rss
                state["best_payload"] = {
                    "rates": rates,
                    "tau_d_ms": tau_d_ms,
                    "alpha": alpha,
                    "baseline": baseline,
                    "scale": scale,
                    "predicted": predicted,
                    "residuals": residuals,
                    "rss": rss,
                }

            now = time.perf_counter()

            if heartbeat_seconds > 0 and now - state["last_heartbeat"] >= heartbeat_seconds:
                elapsed = now - start_time
                print(
                    f"[surface_fit:{model_name}] "
                    f"nfev={state['nfev']} "
                    f"elapsed={elapsed:.1f}s "
                    f"best_rss={state['best_rss']:.6g} "
                    f"tau_d_ms={tau_d_ms:.6g} "
                    f"alpha={alpha:.6g}",
                    flush=True,
                )
                state["last_heartbeat"] = now

            return residuals[finite_mask]

        except Exception as exc:

            state["last_error_type"] = type(exc).__name__
            state["last_error_message"] = str(exc)
            state["last_traceback"] = traceback.format_exc()

            now = time.perf_counter()

            if heartbeat_seconds > 0 and now - state["last_heartbeat"] >= heartbeat_seconds:
                elapsed = now - start_time
                print(
                    f"[surface_fit:{model_name}] "
                    f"nfev={state['nfev']} "
                    f"elapsed={elapsed:.1f}s "
                    f"last_error={type(exc).__name__}: {exc}",
                    flush=True,
                )
                state["last_heartbeat"] = now

            return make_large_residual(observed_matrix)

    result = least_squares(
        residual_function,
        x0=x0,
        bounds=(lower, upper),
        method="trf",
        max_nfev=max_nfev,
        xtol=rtol,
        ftol=rtol,
        gtol=rtol,
    )

    # Re-evaluate once at optimum to ensure payload corresponds to final point.
    residual_function(result.x)

    if state["best_payload"] is None:
        details = (
            f"{state['last_error_type']}: {state['last_error_message']}"
            if state["last_error_type"]
            else "unknown error"
        )

        raise RuntimeError(
            f"Surface fit failed for {model_name}: no valid evaluation. "
            f"Last objective error was {details}.\n"
            f"{state['last_traceback'] or ''}"
        )

    payload = state["best_payload"]

    n_observations = int(finite_mask.sum())
    n_parameters = len(parameter_names) + 2 + 2
    # ODE params + tau_d/alpha + linear baseline/scale.

    rmse, aic, bic = calculate_statistics(
        rss=payload["rss"],
        n_observations=n_observations,
        n_parameters=n_parameters,
    )

    model_output_dir = output_dir / "models" / safe_name(model_name)
    model_output_dir.mkdir(parents=True, exist_ok=True)

    predicted_dataframe = pd.DataFrame(
        payload["predicted"],
        columns=[
            f"G_tau_{tau:.8g}_ms"
            for tau in tau_values
        ],
    )
    predicted_dataframe.insert(0, "time_min", time_values)

    residuals_dataframe = pd.DataFrame(
        payload["residuals"],
        columns=[
            f"G_tau_{tau:.8g}_ms"
            for tau in tau_values
        ],
    )
    residuals_dataframe.insert(0, "time_min", time_values)

    predicted_dataframe.to_csv(model_output_dir / "surface_predicted.csv", index=False)
    residuals_dataframe.to_csv(model_output_dir / "surface_residuals.csv", index=False)

    tau_map = pd.DataFrame(
        {
            "column": [
                f"G_tau_{tau:.8g}_ms"
                for tau in tau_values
            ],
            "tau_ms": tau_values,
        }
    )
    tau_map.to_csv(model_output_dir / "tau_map.csv", index=False)

    fit_result = FCSSurfaceFitResult(
        model_name=model_name,
        success=bool(result.success),
        message=str(result.message),
        nfev=int(result.nfev),
        rss=float(payload["rss"]),
        rmse=float(rmse),
        aic=float(aic),
        bic=float(bic),
        n_observations=n_observations,
        n_parameters=n_parameters,
        baseline=float(payload["baseline"]),
        scale=float(payload["scale"]),
        tau_d_ms=float(payload["tau_d_ms"]),
        alpha=float(payload["alpha"]),
        fitted_parameters={
            key: float(value)
            for key, value in payload["rates"].items()
        },
        output_dir=str(model_output_dir),
    )

    with (model_output_dir / "surface_fit_summary.json").open("w") as handle:
        json.dump(asdict(fit_result), handle, indent=2, default=json_default)

    return fit_result


def compute_weights(table: pd.DataFrame, criterion: str) -> pd.DataFrame:
    output = table.copy()

    if criterion not in output.columns:
        return output

    values = pd.to_numeric(output[criterion], errors="coerce")

    valid = values.notna() & output["success"].astype(bool)

    output[f"delta_{criterion}"] = np.nan
    output[f"{criterion}_weight"] = np.nan
    output[f"{criterion}_weight_percent"] = np.nan

    if not valid.any():
        return output

    delta = values[valid] - values[valid].min()
    raw = np.exp(-0.5 * delta.to_numpy(dtype=float))
    weights = raw / raw.sum()

    output.loc[valid, f"delta_{criterion}"] = delta
    output.loc[valid, f"{criterion}_weight"] = weights
    output.loc[valid, f"{criterion}_weight_percent"] = 100.0 * weights

    return output

def write_surface_model_preview(
    *,
    output_dir: Path,
    model_texts: dict[str, str],
) -> None:
    rows = []

    for model_name, model_text in model_texts.items():
        try:
            model = build_model_spec(
                model_text,
                name=model_name,
            )
            parameter_names = model_parameter_names(model)
            species_names = model_species_names(model)
        except Exception:
            parameter_names = infer_parameter_names(model_text)
            species_names = infer_species_names(model_text)

        rows.append(
            {
                "model_name": model_name,
                "model_text": model_text,
                "parameters": ", ".join(str(name) for name in parameter_names),
                "species": ", ".join(str(name) for name in species_names),
            }
        )

    pd.DataFrame(rows).to_csv(
        output_dir / "surface_model_preview.csv",
        index=False,
    )

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit raw FCS G(tau, time) surfaces to ODE TXT models."
    )

    parser.add_argument("--data", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", default="fcs_ode/fits_fcs_surface_numba")
    parser.add_argument("--engine-name", default="numba_projection")
    parser.add_argument("--time-column", default="time_min")
    parser.add_argument("--observed-species", default=None)
    parser.add_argument("--initial-active-species", default=None)

    parser.add_argument("--initial-guess", type=float, default=0.1)
    parser.add_argument("--lower-bound", type=float, default=1e-8)
    parser.add_argument("--upper-bound", type=float, default=100.0)

    parser.add_argument("--tau-d-initial-ms", type=float, default=None)
    parser.add_argument("--tau-d-lower-ms", type=float, default=1e-6)
    parser.add_argument("--tau-d-upper-ms", type=float, default=1e8)

    parser.add_argument("--alpha-initial", type=float, default=1.0)
    parser.add_argument("--alpha-lower", type=float, default=0.05)
    parser.add_argument("--alpha-upper", type=float, default=5.0)

    parser.add_argument("--max-nfev", type=int, default=300)
    parser.add_argument("--rtol", type=float, default=1e-7)
    parser.add_argument("--atol", type=float, default=1e-10)
    parser.add_argument("--sort-by", default="bic")
    parser.add_argument("--heartbeat-seconds", type=float, default=10.0)
    parser.add_argument("--stop-on-failure", action="store_true")

    args = parser.parse_args()

    data_path = Path(args.data)
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    time_values, signal_columns, tau_values, raw_dataframe, observed_matrix = load_raw_fcs_matrix(
        data_path=data_path,
        time_column=args.time_column,
    )

    model_texts = read_model_files(model_dir)

    write_surface_model_preview(
        output_dir=output_dir,
        model_texts=model_texts,
    )

    run_config = vars(args).copy()
    run_config["data"] = str(data_path)
    run_config["model_dir"] = str(model_dir)
    run_config["output_dir"] = str(output_dir)
    run_config["n_timepoints"] = int(len(time_values))
    run_config["n_tau"] = int(len(tau_values))
    run_config["tau_min_ms"] = float(np.nanmin(tau_values))
    run_config["tau_max_ms"] = float(np.nanmax(tau_values))

    with (output_dir / "surface_fit_run_config.json").open("w") as handle:
        json.dump(run_config, handle, indent=2, default=json_default)

    pd.DataFrame(
        {
            "column": signal_columns,
            "tau_ms": tau_values,
        }
    ).to_csv(output_dir / "surface_tau_column_map.csv", index=False)

    print("\nRaw FCS surface fitting")
    print("=======================")
    print(f"Data: {data_path}")
    print(f"Model dir: {model_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Engine: {args.engine_name}")
    print(f"Timepoints: {len(time_values)}")
    print(f"Tau points: {len(tau_values)}")
    print(f"Models: {len(model_texts)}")
    print(f"Surface shape: {observed_matrix.shape}")

    tracker = ProgressTracker(
        stage="fcs_surface_model_comparison",
        total=len(model_texts),
        callback=lambda event: print(event.to_console_line(), flush=True),
        payload={
            "engine_name": args.engine_name,
            "time_column": args.time_column,
        },
    )

    results: list[FCSSurfaceFitResult] = []
    failures = []

    tracker.emit(
        current=0,
        message=f"Starting FCS surface comparison over {len(model_texts)} models",
    )

    for index, (model_name, model_text) in enumerate(model_texts.items(), start=1):
        tracker.emit(
            current=index - 1,
            message=f"Fitting surface model {model_name}",
            payload={"model_name": model_name},
        )

        try:
            fit_result = fit_one_surface_model(
                model_name=model_name,
                model_text=model_text,
                time_values=time_values,
                tau_values=tau_values,
                observed_matrix=observed_matrix,
                output_dir=output_dir,
                engine_name=args.engine_name,
                observed_species=args.observed_species,
                active_species=args.initial_active_species,
                initial_guess=args.initial_guess,
                lower_bound=args.lower_bound,
                upper_bound=args.upper_bound,
                tau_d_initial_ms=args.tau_d_initial_ms,
                tau_d_lower_ms=args.tau_d_lower_ms,
                tau_d_upper_ms=args.tau_d_upper_ms,
                alpha_initial=args.alpha_initial,
                alpha_lower=args.alpha_lower,
                alpha_upper=args.alpha_upper,
                max_nfev=args.max_nfev,
                rtol=args.rtol,
                atol=args.atol,
                heartbeat_seconds=args.heartbeat_seconds,
            )

            results.append(fit_result)

            tracker.emit(
                current=index,
                message=f"Finished surface model {model_name}",
                payload={
                    "model_name": model_name,
                    "rss": fit_result.rss,
                    "bic": fit_result.bic,
                    "success": fit_result.success,
                },
            )

        except Exception as exc:
            failure = {
                "model_name": model_name,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            }
            failures.append(failure)

            tracker.emit(
                current=index,
                message=f"Failed surface model {model_name}",
                payload=failure,
            )

            if args.stop_on_failure:
                raise

    table = pd.DataFrame([asdict(result) for result in results])

    if not table.empty:
        table = compute_weights(table, "bic")
        table = compute_weights(table, "aic")

        if args.sort_by in table.columns:
            table = table.sort_values(args.sort_by, na_position="last")

        table = table.reset_index(drop=True)
        table.insert(0, "rank", range(1, len(table) + 1))

    table_path = output_dir / "fcs_surface_model_comparison_table.csv"
    table.to_csv(table_path, index=False)

    failures_path = output_dir / "fcs_surface_model_comparison_failures.json"
    with failures_path.open("w") as handle:
        json.dump(failures, handle, indent=2)

    summary_path = output_dir / "fcs_surface_model_comparison_summary.json"
    with summary_path.open("w") as handle:
        json.dump(
            {
                "comparison_table": table.to_dict(orient="records"),
                "failures": failures,
                "sort_by": args.sort_by,
            },
            handle,
            indent=2,
            default=json_default,
        )

    tracker.emit(
        current=len(model_texts),
        message="Finished FCS surface model comparison",
        payload={
            "n_models": len(model_texts),
            "n_successful": int(len(results)),
            "n_failed": int(len(failures)),
        },
    )

    print("\nFCS surface model comparison complete")
    print("=====================================")

    if not table.empty:
        display_columns = [
            column
            for column in [
                "rank",
                "model_name",
                "success",
                "rss",
                "rmse",
                "aic",
                "bic",
                "bic_weight_percent",
                "tau_d_ms",
                "alpha",
                "baseline",
                "scale",
                "nfev",
            ]
            if column in table.columns
        ]

        print(table[display_columns].to_string(index=False))
    else:
        print("No successful fits.")

    print("\nWritten files:")
    print(f"  table: {table_path}")
    print(f"  failures: {failures_path}")
    print(f"  summary: {summary_path}")
    print(f"  models: {output_dir / 'models'}")


if __name__ == "__main__":
    main()
