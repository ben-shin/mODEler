#!/usr/bin/env python3
"""
Batch runner for mODEler FCS ODE model fitting.

Designed for running every .txt model file in the current fcs_ode directory
against modeler_by_time_10_to_220.csv.

Typical use from ~/mODEler/fcs_ode:

    python run_modeler_10_to_220.py

Quick dry run, no fitting:

    python run_modeler_10_to_220.py --dry-run

Overwrite existing results:

    python run_modeler_10_to_220.py --overwrite

Use fewer starts for testing:

    python run_modeler_10_to_220.py --n-starts 5 --n-workers 1

Variable projection is ON by default. To disable it:

    python run_modeler_10_to_220.py --no-variable-projection
"""

from __future__ import annotations

import argparse
import json
import queue
import re
import subprocess
import threading
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from tqdm import tqdm
except ImportError:  # tqdm is optional
    tqdm = None


# -----------------------------------------------------------------------------
# Model parsing helpers
# -----------------------------------------------------------------------------


def strip_comment(line: str) -> str:
    return line.split("#", 1)[0].strip()


def parse_side(side: str) -> dict[str, float]:
    """
    Parse one side of a compact mODEler reaction.

    Examples
    --------
    P1        -> {"P1": 1}
    2P1       -> {"P1": 2}
    P1 + P2   -> {"P1": 1, "P2": 1}
    """
    stoich: dict[str, float] = {}

    for raw_term in side.split("+"):
        term = raw_term.strip()
        if not term:
            continue

        match = re.match(
            r"^\s*(?:(\d+(?:\.\d+)?)\s*)?([A-Za-z_][A-Za-z0-9_]*)\s*$",
            term,
        )
        if match is None:
            raise ValueError(f"Could not parse reaction term: {term!r}")

        coeff_text, species = match.groups()
        coeff = float(coeff_text) if coeff_text is not None else 1.0
        stoich[species] = stoich.get(species, 0.0) + coeff

    return stoich


def parse_model_file(model_path: Path) -> tuple[list[str], list[dict[str, Any]], list[str], list[str]]:
    """
    Infer reactions, species, and mODEler parameter names from a compact model txt.

    Assumptions
    -----------
    line 1 irreversible A>B gives k1f
    line 1 reversible   A-B gives k1f and k1r
    line 2 gives k2f/k2r, etc.
    """
    reactions: list[str] = []
    species: list[str] = []

    for raw in model_path.read_text().splitlines():
        line = strip_comment(raw)
        if line:
            reactions.append(line)

    if not reactions:
        raise ValueError(f"No reactions found in {model_path}")

    parameters: list[str] = []
    parsed_reactions: list[dict[str, Any]] = []

    for i, rxn in enumerate(reactions, start=1):
        if ">" in rxn:
            left, right = rxn.split(">", 1)
            reversible = False
            parameters.append(f"k{i}f")
        elif "-" in rxn:
            left, right = rxn.split("-", 1)
            reversible = True
            parameters.append(f"k{i}f")
            parameters.append(f"k{i}r")
        else:
            raise ValueError(
                f"Cannot parse reaction line in {model_path}: {rxn!r}. "
                "Expected irreversible A>B or reversible A-B syntax."
            )

        left_stoich = parse_side(left)
        right_stoich = parse_side(right)

        for s in list(left_stoich.keys()) + list(right_stoich.keys()):
            if s not in species:
                species.append(s)

        parsed_reactions.append(
            {
                "line_number": i,
                "reaction": rxn,
                "left": left_stoich,
                "right": right_stoich,
                "reversible": reversible,
            }
        )

    return reactions, parsed_reactions, species, parameters


def species_monomer_equivalent(species: str) -> int | None:
    """For species named P1, P2, P12, return 1, 2, 12. Otherwise return None."""
    match = re.fullmatch(r"P(\d+)", species)
    if match is None:
        return None
    return int(match.group(1))


def check_mass_balance(parsed_reactions: list[dict[str, Any]]) -> list[str]:
    """
    Check mass conservation for Pn-style species.

    Example
    -------
    2P1-P2 is balanced because left mass = 2*1 and right mass = 1*2.
    """
    warnings: list[str] = []

    for rxn in parsed_reactions:
        all_species = list(rxn["left"].keys()) + list(rxn["right"].keys())
        sizes = {s: species_monomer_equivalent(s) for s in all_species}

        if any(v is None for v in sizes.values()):
            warnings.append(
                f"reaction {rxn['line_number']} not checked because not all species are Pn-style: "
                f"{rxn['reaction']}"
            )
            continue

        left_mass = sum(coeff * sizes[s] for s, coeff in rxn["left"].items())
        right_mass = sum(coeff * sizes[s] for s, coeff in rxn["right"].items())

        if abs(left_mass - right_mass) > 1e-12:
            warnings.append(
                f"reaction {rxn['line_number']} is NOT mass balanced: "
                f"{rxn['reaction']} | left mass={left_mass}, right mass={right_mass}"
            )

    return warnings


# -----------------------------------------------------------------------------
# Config generation
# -----------------------------------------------------------------------------


def build_config(
    *,
    model_path: Path,
    data_path: Path,
    time_column: str,
    observed_species: str,
    use_variable_projection: bool,
    variable_projection_backend: str,
    variable_projection_method: str,
    species: list[str],
    parameters: list[str],
    output_dir: Path,
    initial_species: str,
    initial_conc: float,
    rate_guess: float,
    rate_lower: float,
    rate_upper: float,
    scale_lower: float,
    scale_upper: float,
    offset_lower: float,
    offset_upper: float,
    max_nfev: int,
    no_plots: bool,
) -> dict[str, Any]:
    if initial_species not in species:
        raise ValueError(
            f"Initial species {initial_species!r} is not present in {model_path.name}. "
            f"Detected species: {species}"
        )

    if observed_species not in species:
        raise ValueError(
            f"Observed species {observed_species!r} is not present in {model_path.name}. "
            f"Detected species: {species}"
        )

    config: dict[str, Any] = {
        "model": str(model_path),
        "data": str(data_path),
        "time_column": time_column,
        "observed_species": observed_species,
        "use_variable_projection": use_variable_projection,
        "variable_projection_backend": variable_projection_backend,
        "variable_projection_method": variable_projection_method,
        "parameters": {
            parameter: {
                "initial_guess": rate_guess,
                "lower_bound": rate_lower,
                "upper_bound": rate_upper,
            }
            for parameter in parameters
        },
        "initial_conditions": {},
        "fit_scale": True,
        "fit_offset": True,
        "scale_initial_guess": 1.0,
        "scale_lower_bound": scale_lower,
        "scale_upper_bound": scale_upper,
        "offset_initial_guess": 0.0,
        "offset_lower_bound": offset_lower,
        "offset_upper_bound": offset_upper,
        "method": "trf",
        "loss": "linear",
        "rtol": 1e-8,
        "atol": 1e-10,
        "max_nfev": max_nfev,
        "output_dir": str(output_dir),
        "no_plots": no_plots,
    }

    for s in species:
        value = initial_conc if s == initial_species else 0.0
        config["initial_conditions"][s] = {
            "value": value,
            "mode": "fixed",
            "lower_bound": 0.0,
            "upper_bound": initial_conc,
        }

    return config


# -----------------------------------------------------------------------------
# Run/result helpers
# -----------------------------------------------------------------------------


def format_seconds(seconds: float | None) -> str:
    if seconds is None or pd.isna(seconds):
        return "unknown"
    return str(timedelta(seconds=int(round(seconds))))


def read_data_header(data_path: Path, requested_time_column: str | None) -> tuple[str, list[str]]:
    df_head = pd.read_csv(data_path, nrows=5)
    columns = list(df_head.columns)

    if requested_time_column is not None:
        time_column = requested_time_column
    elif "time_min" in columns:
        time_column = "time_min"
    elif "time" in columns:
        time_column = "time"
    else:
        time_column = columns[0]

    if time_column not in columns:
        raise ValueError(
            f"Time column {time_column!r} not found in {data_path}. "
            f"Available columns: {columns}"
        )

    return time_column, columns


def find_comparison_files(output_dir: Path) -> list[Path]:
    if not output_dir.exists():
        return []

    candidates: list[Path] = []
    preferred_names = [
        "parallel_multistart_comparison.csv",
        "multistart_comparison.csv",
        "comparison.csv",
        "variable_projection_multistart_comparison.csv",
    ]

    for name in preferred_names:
        path = output_dir / name
        if path.exists():
            candidates.append(path)

    for path in output_dir.rglob("*.csv"):
        lower = path.name.lower()
        if "comparison" in lower and path not in candidates:
            candidates.append(path)

    return candidates


def best_row_from_comparison(comparison_csv: Path) -> dict[str, Any]:
    try:
        df = pd.read_csv(comparison_csv)
    except Exception as exc:
        return {
            "comparison_file": str(comparison_csv),
            "comparison_read_error": str(exc),
        }

    if df.empty:
        return {
            "comparison_file": str(comparison_csv),
            "comparison_empty": True,
        }

    lower_cols = {c.lower(): c for c in df.columns}
    sort_col = None

    for candidate in ["bic", "aic", "rmse", "rss", "cost"]:
        if candidate in lower_cols:
            sort_col = lower_cols[candidate]
            break

    if sort_col is not None:
        best = df.sort_values(sort_col, ascending=True).iloc[0].to_dict()
    else:
        best = df.iloc[0].to_dict()

    best["comparison_file"] = str(comparison_csv)
    best["sort_col"] = sort_col or ""
    return best


def run_subprocess_with_log(
    cmd: list[str],
    log_path: Path,
    quiet: bool,
    heartbeat_seconds: int,
    model_name: str,
    status_path: Path | None = None,
    n_starts: int | None = None,
    n_workers: int | None = None,
) -> tuple[int, float]:
    """Run a subprocess while writing a log and printing periodic heartbeat updates.

    mODEler's variable-projection multistart command may not print per-start progress
    while worker processes are busy. This heartbeat proves the process is still alive
    and records useful timing information even when stdout is quiet.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    heartbeat_seconds = max(1, int(heartbeat_seconds))

    output_queue: queue.Queue[str | None] = queue.Queue()

    def reader_thread(stream) -> None:
        try:
            for line in stream:
                output_queue.put(line)
        finally:
            output_queue.put(None)

    def drain_output(log_handle) -> bool:
        """Drain available subprocess output. Return True once reader is finished."""
        reader_finished = False
        while True:
            try:
                item = output_queue.get_nowait()
            except queue.Empty:
                break
            if item is None:
                reader_finished = True
                continue
            log_handle.write(item)
            log_handle.flush()
            if not quiet:
                print(item, end="", flush=True)
        return reader_finished

    with log_path.open("w") as log:
        log.write(f"Started: {datetime.now().isoformat(timespec='seconds')}\n")
        log.write("Command:\n")
        log.write(" ".join(cmd) + "\n\n")
        log.flush()

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None

        thread = threading.Thread(target=reader_thread, args=(process.stdout,), daemon=True)
        thread.start()

        last_heartbeat = 0.0
        reader_finished = False

        while process.poll() is None:
            reader_finished = drain_output(log) or reader_finished
            elapsed = time.time() - start

            if elapsed - last_heartbeat >= heartbeat_seconds:
                last_heartbeat = elapsed
                message = (
                    f"[heartbeat] {model_name} still running | "
                    f"elapsed={format_seconds(elapsed)}"
                )
                if n_starts is not None:
                    message += f" | starts={n_starts}"
                if n_workers is not None:
                    message += f" | workers={n_workers}"
                message += f" | log={log_path}"

                print(message, flush=True)
                log.write(message + "\n")
                log.flush()

                if status_path is not None:
                    write_status(
                        status_path,
                        {
                            "last_updated": datetime.now().isoformat(timespec="seconds"),
                            "current_model": model_name,
                            "state": "running_subprocess",
                            "elapsed_seconds_current_model": elapsed,
                            "elapsed_current_model_hms": format_seconds(elapsed),
                            "n_starts": n_starts,
                            "n_workers": n_workers,
                            "log_file": str(log_path),
                            "command": " ".join(cmd),
                            "note": "Heartbeat only. Per-start progress is only available if mODEler emits it.",
                        },
                    )

            time.sleep(1.0)

        return_code = process.wait()

        # Drain anything printed right before process exit.
        while not reader_finished:
            reader_finished = drain_output(log) or reader_finished
            if not thread.is_alive() and output_queue.empty():
                break
            time.sleep(0.1)

        elapsed = time.time() - start

        log.write("\n")
        log.write(f"Finished: {datetime.now().isoformat(timespec='seconds')}\n")
        log.write(f"Return code: {return_code}\n")
        log.write(f"Elapsed seconds: {elapsed:.3f}\n")
        log.flush()

    return return_code, elapsed


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def write_status(path: Path, status: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(status, f, indent=2)


def model_looks_complete(output_dir: Path) -> bool:
    return bool(find_comparison_files(output_dir))


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all .txt mODEler models against modeler_by_time_10_to_220.csv."
    )

    parser.add_argument("--model-dir", default=".", help="Directory containing model .txt files. Default: current directory.")
    parser.add_argument("--data", default="modeler_by_time_10_to_220.csv", help="Input CSV file.")
    parser.add_argument("--models-glob", default="*.txt", help="Glob for model files. Default: *.txt")
    parser.add_argument("--config-dir", default="configs_P1_100uM_10_to_220_runner", help="Where to write generated JSON configs.")
    parser.add_argument("--out-root", default="fits_P1_100uM_10_to_220_runner", help="Root folder for fit outputs.")

    parser.add_argument("--time-column", default=None, help="Time column name. Default: auto-detect time_min, time, or first column.")
    parser.add_argument("--initial-species", default="P1", help="Species with nonzero initial concentration. Default: P1")
    parser.add_argument("--observed-species", default="P1", help="Species used as shared observable. Default: P1")
    parser.add_argument("--initial-conc", type=float, default=100.0, help="Initial concentration in uM. Default: 100.0")

    parser.add_argument("--n-starts", type=int, default=50, help="Number of multistart fits per model. Default: 50")
    parser.add_argument("--n-workers", type=int, default=8, help="Workers within each multistart run. Default: 8")
    parser.add_argument("--max-nfev", type=int, default=10000, help="Max optimizer evaluations. Default: 10000")

    parser.add_argument("--rate-guess", type=float, default=1e-7, help="Initial guess for each rate constant. Default: 1e-7")
    parser.add_argument("--rate-lower", type=float, default=1e-15, help="Lower bound for rate constants. Default: 1e-15")
    parser.add_argument("--rate-upper", type=float, default=1e-1, help="Upper bound for rate constants. Default: 1e-1")

    parser.add_argument("--scale-lower", type=float, default=-1_000_000.0, help="Lower bound for per-column scale.")
    parser.add_argument("--scale-upper", type=float, default=1_000_000.0, help="Upper bound for per-column scale.")
    parser.add_argument("--offset-lower", type=float, default=-1_000_000.0, help="Lower bound for per-column offset.")
    parser.add_argument("--offset-upper", type=float, default=1_000_000.0, help="Upper bound for per-column offset.")

    parser.add_argument("--plots", action="store_true", help="Enable mODEler plots. Default: plots disabled.")
    parser.add_argument("--no-variable-projection", action="store_true", help="Disable variable projection. Default: variable projection enabled.")
    parser.add_argument("--variable-projection-backend", default="numpy", help="Variable projection backend. Default: numpy")
    parser.add_argument("--variable-projection-method", default="LSODA", help="ODE method used inside variable projection. Default: LSODA")
    parser.add_argument("--dry-run", action="store_true", help="Generate configs and summaries but do not fit.")
    parser.add_argument("--overwrite", action="store_true", help="Re-run models even if comparison CSVs already exist.")
    parser.add_argument("--skip-missing-species", action="store_true", help="Skip models that do not contain required species instead of erroring.")
    parser.add_argument("--quiet-subprocess", action="store_true", help="Save mODEler output to run.log without streaming it to terminal.")
    parser.add_argument("--heartbeat-seconds", type=int, default=30, help="Print a heartbeat while each model is running. Default: 30 seconds.")

    args = parser.parse_args()

    model_dir = Path(args.model_dir).resolve()
    data_path = Path(args.data).resolve()
    config_dir = Path(args.config_dir).resolve()
    out_root = Path(args.out_root).resolve()

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")
    if not data_path.exists():
        raise FileNotFoundError(f"Data CSV does not exist: {data_path}")

    config_dir.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)

    time_column, data_columns = read_data_header(data_path, args.time_column)
    signal_columns = [c for c in data_columns if c != time_column]

    model_files = sorted(
        p for p in model_dir.glob(args.models_glob)
        if p.is_file() and p.name != "requirements.txt" and not p.name.startswith(".")
    )

    if not model_files:
        raise RuntimeError(f"No model files found: {model_dir / args.models_glob}")

    batch_start = time.time()
    progress_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    model_info_rows: list[dict[str, Any]] = []

    progress_csv = out_root / "progress.csv"
    results_csv = out_root / "all_model_comparison.csv"
    model_info_csv = out_root / "generated_model_info.csv"
    status_json = out_root / "status.json"

    print("=" * 88)
    print("mODEler batch runner: global observable multistart fitting")
    print("=" * 88)
    print(f"Model directory:     {model_dir}")
    print(f"Model glob:          {args.models_glob}")
    print(f"Data CSV:            {data_path}")
    print(f"Time column:         {time_column}")
    print(f"Signal columns:      {len(signal_columns)}")
    print(f"Initial species:     {args.initial_species}")
    print(f"Observed species:    {args.observed_species}")
    print(f"Initial conc:        {args.initial_conc}")
    print(f"Model count:         {len(model_files)}")
    print(f"Variable projection: {not args.no_variable_projection}")
    print(f"VP backend:          {args.variable_projection_backend}")
    print(f"VP ODE method:       {args.variable_projection_method}")
    print(f"Starts per model:    {args.n_starts}")
    print(f"Workers per model:   {args.n_workers}")
    print(f"Heartbeat seconds:   {args.heartbeat_seconds}")
    print(f"Config dir:          {config_dir}")
    print(f"Output root:         {out_root}")
    print(f"Dry run:             {args.dry_run}")
    print(f"Overwrite:           {args.overwrite}")
    print("=" * 88)

    iterable = tqdm(model_files, desc="Models", unit="model") if tqdm is not None else model_files

    for model_index, model_path in enumerate(iterable, start=1):
        model_name = model_path.stem
        output_dir = out_root / model_name
        config_path = config_dir / f"{model_name}_global_multistart.json"
        log_path = output_dir / "run.log"
        model_start = time.time()

        completed = len(progress_rows)
        finished_times = [r["elapsed_seconds"] for r in progress_rows if r.get("elapsed_seconds") is not None]
        avg_time = sum(finished_times) / len(finished_times) if finished_times else None
        remaining = len(model_files) - completed
        eta = avg_time * remaining if avg_time is not None else None

        write_status(
            status_json,
            {
                "batch_started": datetime.fromtimestamp(batch_start).isoformat(timespec="seconds"),
                "last_updated": datetime.now().isoformat(timespec="seconds"),
                "current_model_index": model_index,
                "total_models": len(model_files),
                "current_model": model_name,
                "completed_models": completed,
                "remaining_models": remaining,
                "elapsed_batch_hms": format_seconds(time.time() - batch_start),
                "estimated_remaining_hms": format_seconds(eta),
                "current_output_dir": str(output_dir),
                "current_log": str(log_path),
            },
        )

        print("\n" + "=" * 88)
        print(f"[{model_index}/{len(model_files)}] {model_name}")
        print("=" * 88)
        print(f"Model file: {model_path}")
        print(f"Output dir: {output_dir}")
        print(f"Log file:   {log_path}")

        try:
            if not args.overwrite and model_looks_complete(output_dir):
                elapsed = time.time() - model_start
                row = {
                    "model": model_name,
                    "status": "skipped_existing",
                    "message": "Comparison CSV already exists. Use --overwrite to rerun.",
                    "elapsed_seconds": elapsed,
                    "elapsed_hms": format_seconds(elapsed),
                    "output_dir": str(output_dir),
                    "log_file": str(log_path),
                }
                progress_rows.append(row)
                result_rows.append(row)
                print("Skipping: existing comparison CSV found. Use --overwrite to rerun.")
                write_csv(progress_csv, progress_rows)
                continue

            reactions, parsed_reactions, species, parameters = parse_model_file(model_path)
            mass_warnings = check_mass_balance(parsed_reactions)

            missing_species = args.initial_species not in species or args.observed_species not in species
            if missing_species:
                msg = (
                    f"Required species missing. initial_species={args.initial_species!r}, "
                    f"observed_species={args.observed_species!r}, detected species={species}"
                )
                if args.skip_missing_species:
                    elapsed = time.time() - model_start
                    row = {
                        "model": model_name,
                        "status": "skipped_missing_species",
                        "message": msg,
                        "elapsed_seconds": elapsed,
                        "elapsed_hms": format_seconds(elapsed),
                        "model_file": str(model_path),
                    }
                    progress_rows.append(row)
                    result_rows.append(row)
                    print(f"Skipping: {msg}")
                    write_csv(progress_csv, progress_rows)
                    continue
                raise ValueError(msg)

            config = build_config(
                model_path=model_path,
                data_path=data_path,
                time_column=time_column,
                observed_species=args.observed_species,
                use_variable_projection=not args.no_variable_projection,
                variable_projection_backend=args.variable_projection_backend,
                variable_projection_method=args.variable_projection_method,
                species=species,
                parameters=parameters,
                output_dir=output_dir,
                initial_species=args.initial_species,
                initial_conc=args.initial_conc,
                rate_guess=args.rate_guess,
                rate_lower=args.rate_lower,
                rate_upper=args.rate_upper,
                scale_lower=args.scale_lower,
                scale_upper=args.scale_upper,
                offset_lower=args.offset_lower,
                offset_upper=args.offset_upper,
                max_nfev=args.max_nfev,
                no_plots=not args.plots,
            )
            config_path.write_text(json.dumps(config, indent=2))

            model_info_row = {
                "model": model_name,
                "model_file": str(model_path),
                "config_file": str(config_path),
                "output_dir": str(output_dir),
                "initial_species": args.initial_species,
                "observed_species": args.observed_species,
                "initial_conc": args.initial_conc,
                "n_reactions": len(reactions),
                "n_species": len(species),
                "n_parameters": len(parameters),
                "species": ";".join(species),
                "parameters": ";".join(parameters),
                "mass_balance_warnings": " | ".join(mass_warnings),
            }
            model_info_rows.append(model_info_row)
            write_csv(model_info_csv, model_info_rows)

            print(f"Species:    {', '.join(species)}")
            print(f"Parameters: {', '.join(parameters)}")
            print(f"Reactions:  {len(reactions)}")
            print(f"Config:     {config_path}")
            if mass_warnings:
                print("Mass-balance warnings:")
                for warning in mass_warnings:
                    print(f"  - {warning}")
            else:
                print("Mass balance: OK for Pn-style species")

            if args.dry_run:
                elapsed = time.time() - model_start
                row = {
                    "model": model_name,
                    "status": "dry_run_config_generated",
                    "elapsed_seconds": elapsed,
                    "elapsed_hms": format_seconds(elapsed),
                    "output_dir": str(output_dir),
                    "config_file": str(config_path),
                    "log_file": str(log_path),
                }
                progress_rows.append(row)
                result_rows.append(row)
                write_csv(progress_csv, progress_rows)
                print("Dry run: generated config only.")
                continue

            cmd = [
                sys.executable,
                "-m",
                "odefit.cli",
                "multistart-global-observables",
                "--config",
                str(config_path),
                "--n-workers",
                str(args.n_workers),
                "--n-starts",
                str(args.n_starts),
            ]

            if not args.no_variable_projection:
                cmd.append("--variable-projection")

            print("Running:")
            print(" ".join(cmd))
            print()

            return_code, fit_elapsed = run_subprocess_with_log(
                cmd=cmd,
                log_path=log_path,
                quiet=args.quiet_subprocess,
                heartbeat_seconds=args.heartbeat_seconds,
                model_name=model_name,
                status_path=status_json,
                n_starts=args.n_starts,
                n_workers=args.n_workers,
            )

            elapsed = time.time() - model_start
            progress_row = {
                "model": model_name,
                "status": "ok" if return_code == 0 else "failed",
                "return_code": return_code,
                "started": datetime.fromtimestamp(model_start).isoformat(timespec="seconds"),
                "finished": datetime.now().isoformat(timespec="seconds"),
                "elapsed_seconds": elapsed,
                "elapsed_hms": format_seconds(elapsed),
                "fit_subprocess_seconds": fit_elapsed,
                "fit_subprocess_hms": format_seconds(fit_elapsed),
                "output_dir": str(output_dir),
                "config_file": str(config_path),
                "log_file": str(log_path),
                "n_reactions": len(reactions),
                "n_species": len(species),
                "n_parameters": len(parameters),
            }

            if return_code != 0:
                progress_rows.append(progress_row)
                result_rows.append(progress_row)
                write_csv(progress_csv, progress_rows)
                write_csv(results_csv, result_rows)
                print(f"FAILED after {format_seconds(elapsed)}. See {log_path}")
                continue

            comparison_files = find_comparison_files(output_dir)
            if comparison_files:
                best = best_row_from_comparison(comparison_files[0])
                best.update(progress_row)
                best["comparison_file"] = str(comparison_files[0])
                result_rows.append(best)
                print(f"Finished OK after {format_seconds(elapsed)}")
                print(f"Comparison file: {comparison_files[0]}")
            else:
                progress_row["status"] = "ok_no_comparison_file_found"
                result_rows.append(progress_row)
                print(f"Finished OK after {format_seconds(elapsed)}, but no comparison CSV was found.")

            progress_rows.append(progress_row)
            write_csv(progress_csv, progress_rows)
            write_csv(results_csv, result_rows)

        except Exception as exc:
            elapsed = time.time() - model_start
            row = {
                "model": model_name,
                "status": "setup_error",
                "error": str(exc),
                "started": datetime.fromtimestamp(model_start).isoformat(timespec="seconds"),
                "finished": datetime.now().isoformat(timespec="seconds"),
                "elapsed_seconds": elapsed,
                "elapsed_hms": format_seconds(elapsed),
                "model_file": str(model_path),
                "output_dir": str(output_dir),
                "log_file": str(log_path),
            }
            progress_rows.append(row)
            result_rows.append(row)
            write_csv(progress_csv, progress_rows)
            write_csv(results_csv, result_rows)
            print(f"SETUP ERROR after {format_seconds(elapsed)}: {exc}")

        finally:
            # Keep sorted comparison updated after every model.
            if result_rows:
                results = pd.DataFrame(result_rows)
                for col in ["bic", "BIC", "aic", "AIC", "rmse", "RMSE", "rss", "RSS", "cost", "Cost"]:
                    if col in results.columns:
                        results[col] = pd.to_numeric(results[col], errors="coerce")
                        results = results.sort_values(col, ascending=True, na_position="last")
                        break
                results.to_csv(results_csv, index=False)

    total_elapsed = time.time() - batch_start
    final_status = {
        "batch_started": datetime.fromtimestamp(batch_start).isoformat(timespec="seconds"),
        "batch_finished": datetime.now().isoformat(timespec="seconds"),
        "total_elapsed_seconds": total_elapsed,
        "total_elapsed_hms": format_seconds(total_elapsed),
        "total_models": len(model_files),
        "completed_models": len(progress_rows),
        "progress_csv": str(progress_csv),
        "results_csv": str(results_csv),
        "model_info_csv": str(model_info_csv),
    }
    write_status(status_json, final_status)

    print("\n" + "=" * 88)
    print("Batch fitting complete")
    print("=" * 88)
    print(f"Total runtime: {format_seconds(total_elapsed)}")
    print(f"Progress:      {progress_csv}")
    print(f"Results:       {results_csv}")
    print(f"Model info:    {model_info_csv}")
    print(f"Status:        {status_json}")
    print()
    print("Inspect best-ranked results with:")
    print(f"python - <<'PY'\nimport pandas as pd\ndf = pd.read_csv(r'{results_csv}')\nprint(df.head(20).to_string(index=False))\nPY")


if __name__ == "__main__":
    main()

