#!/usr/bin/env python3

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# ============================================================
# Model parsing
# ============================================================

def strip_comment(line: str) -> str:
    return line.split("#", 1)[0].strip()


def parse_side(side: str):
    stoich = {}

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


def parse_model_file(model_path: Path):
    reactions = []
    species = []

    for raw in model_path.read_text().splitlines():
        line = strip_comment(raw)
        if not line:
            continue
        reactions.append(line)

    if not reactions:
        raise ValueError(f"No reactions found in {model_path}")

    parameters = []
    parsed_reactions = []

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
                "Expected A>B or A-B syntax."
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


def species_monomer_equivalent(species: str):
    match = re.fullmatch(r"P(\d+)", species)
    if match is None:
        return None
    return int(match.group(1))


def check_mass_balance(parsed_reactions):
    warnings = []

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


# ============================================================
# Config generation
# ============================================================

def build_config(
    model_path,
    data_path,
    time_column,
    observed_species,
    species,
    parameters,
    output_dir,
    rate_guess,
    rate_lower,
    rate_upper,
    initial_species,
    initial_conc,
    no_plots,
    max_nfev,
):
    if initial_species not in species:
        raise ValueError(
            f"Initial species {initial_species!r} is not present in model. "
            f"Detected species: {species}"
        )

    if observed_species not in species:
        raise ValueError(
            f"Observed species {observed_species!r} is not present in model. "
            f"Detected species: {species}"
        )

    config = {
        "model": str(model_path),
        "data": str(data_path),
        "time_column": time_column,
        "observed_species": observed_species,

        "parameters": {
            p: {
                "initial_guess": rate_guess,
                "lower_bound": rate_lower,
                "upper_bound": rate_upper,
            }
            for p in parameters
        },

        "initial_conditions": {},

        "fit_scale": True,
        "fit_offset": True,

        "scale_initial_guess": 1.0,
        "scale_lower_bound": -1000000.0,
        "scale_upper_bound": 1000000.0,

        "offset_initial_guess": 0.0,
        "offset_lower_bound": -1000000.0,
        "offset_upper_bound": 1000000.0,

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


# ============================================================
# Results and progress helpers
# ============================================================

def format_seconds(seconds):
    if seconds is None or not pd.notna(seconds):
        return "unknown"

    seconds = int(round(seconds))
    return str(timedelta(seconds=seconds))


def write_progress_files(out_root: Path, progress_rows, current_status):
    progress_csv = out_root / "progress.csv"
    status_json = out_root / "status.json"

    pd.DataFrame(progress_rows).to_csv(progress_csv, index=False)

    with status_json.open("w") as f:
        json.dump(current_status, f, indent=2)


def find_comparison_files(output_dir: Path):
    if not output_dir.exists():
        return []

    candidates = []

    for name in [
        "parallel_multistart_comparison.csv",
        "multistart_comparison.csv",
        "comparison.csv",
    ]:
        path = output_dir / name
        if path.exists():
            candidates.append(path)

    for path in output_dir.rglob("*.csv"):
        lower = path.name.lower()
        if "comparison" in lower and path not in candidates:
            candidates.append(path)

    return candidates


def best_row_from_comparison(comparison_csv: Path):
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
    best["sort_col"] = sort_col if sort_col is not None else ""
    return best


def run_subprocess_with_log(cmd, log_path: Path, quiet=False):
    log_path.parent.mkdir(parents=True, exist_ok=True)

    start = time.time()

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

        for line in process.stdout:
            log.write(line)
            log.flush()

            if not quiet:
                print(line, end="")

        return_code = process.wait()

        elapsed = time.time() - start
        log.write("\n")
        log.write(f"Finished: {datetime.now().isoformat(timespec='seconds')}\n")
        log.write(f"Return code: {return_code}\n")
        log.write(f"Elapsed seconds: {elapsed:.3f}\n")
        log.flush()

    return return_code, elapsed


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run all mODEler global-observable ODE fits with progress bars, "
            "ETA estimates, per-model logs, progress.csv, and status.json."
        )
    )

    parser.add_argument("--model-dir", default="fcs_ode")
    parser.add_argument("--data", default="fcs_ode/modeler_by_time_clean.csv")
    parser.add_argument("--config-dir", default="fcs_ode/configs_P1_100uM_progress")
    parser.add_argument("--out-root", default="fcs_ode/fits_P1_100uM_progress")

    parser.add_argument("--time-column", default=None)
    parser.add_argument("--initial-species", default="P1")
    parser.add_argument("--initial-conc", type=float, default=100.0)
    parser.add_argument("--observed-species", default="P1")

    parser.add_argument("--n-starts", type=int, default=50)
    parser.add_argument("--n-workers", type=int, default=8)

    parser.add_argument("--rate-guess", type=float, default=1e-7)
    parser.add_argument("--rate-lower", type=float, default=1e-15)
    parser.add_argument("--rate-upper", type=float, default=1e-1)
    parser.add_argument("--max-nfev", type=int, default=10000)

    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-missing-species", action="store_true")

    parser.add_argument(
        "--quiet-subprocess",
        action="store_true",
        help="Do not stream mODEler output to terminal; save it only to run.log.",
    )

    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    data_path = Path(args.data)
    config_dir = Path(args.config_dir)
    out_root = Path(args.out_root)

    config_dir.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)

    if not model_dir.exists():
        raise FileNotFoundError(f"Could not find model directory: {model_dir}")

    if not data_path.exists():
        raise FileNotFoundError(f"Could not find data file: {data_path}")

    df_head = pd.read_csv(data_path, nrows=5)

    if args.time_column is not None:
        time_column = args.time_column
    elif "time_min" in df_head.columns:
        time_column = "time_min"
    elif "time" in df_head.columns:
        time_column = "time"
    else:
        time_column = df_head.columns[0]

    model_files = sorted(
        p for p in model_dir.glob("*.txt")
        if p.name != "requirements.txt"
    )

    if not model_files:
        raise RuntimeError(f"No model .txt files found in {model_dir}")

    print("=" * 80)
    print("mODEler global observable batch fitting with progress")
    print("=" * 80)
    print(f"Model directory:        {model_dir}")
    print(f"Data file:              {data_path}")
    print(f"Time column:            {time_column}")
    print(f"Initial species:        {args.initial_species}")
    print(f"Initial concentration:  {args.initial_conc}")
    print(f"Observed species:       {args.observed_species}")
    print(f"Config directory:       {config_dir}")
    print(f"Output root:            {out_root}")
    print(f"Models found:           {len(model_files)}")
    print(f"Starts per model:       {args.n_starts}")
    print(f"Workers per model:      {args.n_workers}")
    print(f"Dry run:                {args.dry_run}")
    print("=" * 80)
    print()

    model_info_rows = []
    result_rows = []
    progress_rows = []

    batch_start = time.time()

    if tqdm is not None:
        model_iter = tqdm(model_files, desc="Models", unit="model")
    else:
        model_iter = model_files

    for model_index, model_path in enumerate(model_iter, start=1):
        model_name = model_path.stem
        output_dir = out_root / model_name
        config_path = config_dir / f"{model_name}_global_multistart.json"
        log_path = output_dir / "run.log"

        model_start = time.time()

        completed_before = len([r for r in progress_rows if r.get("status") in {"ok", "failed", "setup_error", "skipped", "dry_run"}])
        elapsed_batch = time.time() - batch_start

        finished_runtimes = [
            r["elapsed_seconds"]
            for r in progress_rows
            if r.get("elapsed_seconds") is not None and pd.notna(r.get("elapsed_seconds"))
        ]

        avg_runtime = sum(finished_runtimes) / len(finished_runtimes) if finished_runtimes else None
        remaining_models = len(model_files) - completed_before

        eta_seconds = avg_runtime * remaining_models if avg_runtime is not None else None

        current_status = {
            "batch_started": datetime.fromtimestamp(batch_start).isoformat(timespec="seconds"),
            "last_updated": datetime.now().isoformat(timespec="seconds"),
            "current_model_index": model_index,
            "total_models": len(model_files),
            "current_model": model_name,
            "completed_models": completed_before,
            "remaining_models": remaining_models,
            "elapsed_batch_seconds": elapsed_batch,
            "elapsed_batch_hms": format_seconds(elapsed_batch),
            "average_finished_model_seconds": avg_runtime,
            "average_finished_model_hms": format_seconds(avg_runtime),
            "estimated_remaining_seconds": eta_seconds,
            "estimated_remaining_hms": format_seconds(eta_seconds),
            "output_dir": str(output_dir),
            "log_file": str(log_path),
        }

        write_progress_files(out_root, progress_rows, current_status)

        if tqdm is not None:
            model_iter.set_postfix({
                "current": model_name,
                "avg": format_seconds(avg_runtime),
                "ETA": format_seconds(eta_seconds),
            })

        print()
        print("=" * 80)
        print(f"[{model_index}/{len(model_files)}] Model: {model_name}")
        print("=" * 80)
        print(f"Started:        {datetime.now().isoformat(timespec='seconds')}")
        print(f"Estimated ETA:  {format_seconds(eta_seconds)}")
        print(f"Output dir:     {output_dir}")
        print(f"Log file:       {log_path}")

        try:
            reactions, parsed_reactions, species, parameters = parse_model_file(model_path)
            mass_warnings = check_mass_balance(parsed_reactions)

            missing_required = (
                args.initial_species not in species
                or args.observed_species not in species
            )

            if missing_required:
                msg = (
                    f"Required species missing. "
                    f"initial_species={args.initial_species!r}, "
                    f"observed_species={args.observed_species!r}, "
                    f"detected species={species}"
                )

                if args.skip_missing_species:
                    elapsed = time.time() - model_start
                    row = {
                        "model": model_name,
                        "status": "skipped",
                        "message": msg,
                        "started": datetime.fromtimestamp(model_start).isoformat(timespec="seconds"),
                        "finished": datetime.now().isoformat(timespec="seconds"),
                        "elapsed_seconds": elapsed,
                        "elapsed_hms": format_seconds(elapsed),
                        "output_dir": str(output_dir),
                        "log_file": str(log_path),
                    }
                    progress_rows.append(row)
                    result_rows.append(row)
                    print(f"SKIPPED: {msg}")
                    continue

                raise ValueError(msg)

            config = build_config(
                model_path=model_path,
                data_path=data_path,
                time_column=time_column,
                observed_species=args.observed_species,
                species=species,
                parameters=parameters,
                output_dir=output_dir,
                rate_guess=args.rate_guess,
                rate_lower=args.rate_lower,
                rate_upper=args.rate_upper,
                initial_species=args.initial_species,
                initial_conc=args.initial_conc,
                no_plots=not args.plots,
                max_nfev=args.max_nfev,
            )

            config_path.write_text(json.dumps(config, indent=2))

            model_info_rows.append({
                "model": model_name,
                "model_file": str(model_path),
                "config_file": str(config_path),
                "output_dir": str(output_dir),
                "initial_species": args.initial_species,
                "initial_conc": args.initial_conc,
                "observed_species": args.observed_species,
                "species": ";".join(species),
                "parameters": ";".join(parameters),
                "n_species": len(species),
                "n_parameters": len(parameters),
                "n_reactions": len(reactions),
                "mass_balance_warnings": " | ".join(mass_warnings),
            })

            print(f"Species:       {', '.join(species)}")
            print(f"Parameters:    {len(parameters)}")
            print(f"Reactions:     {len(reactions)}")
            print(f"Mass balance:  {'OK' if not mass_warnings else 'WARNINGS'}")

            if args.dry_run:
                elapsed = time.time() - model_start
                row = {
                    "model": model_name,
                    "status": "dry_run",
                    "started": datetime.fromtimestamp(model_start).isoformat(timespec="seconds"),
                    "finished": datetime.now().isoformat(timespec="seconds"),
                    "elapsed_seconds": elapsed,
                    "elapsed_hms": format_seconds(elapsed),
                    "output_dir": str(output_dir),
                    "config_file": str(config_path),
                    "log_file": str(log_path),
                }
                progress_rows.append(row)
                result_rows.append(row)
                write_progress_files(out_root, progress_rows, current_status)
                print("Dry run: config generated, not fitting.")
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

            print("Running:")
            print(" ".join(cmd))
            print()

            return_code, fit_elapsed = run_subprocess_with_log(
                cmd,
                log_path,
                quiet=args.quiet_subprocess,
            )

            elapsed = time.time() - model_start
            status = "ok" if return_code == 0 else "failed"

            progress_row = {
                "model": model_name,
                "status": status,
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
                "n_species": len(species),
                "n_parameters": len(parameters),
                "n_reactions": len(reactions),
            }

            progress_rows.append(progress_row)

            if return_code != 0:
                result_rows.append(progress_row)
                print(f"FAILED after {format_seconds(elapsed)}")
                write_progress_files(out_root, progress_rows, current_status)
                continue

            comparison_files = find_comparison_files(output_dir)

            if comparison_files:
                comparison_file = comparison_files[0]
                best = best_row_from_comparison(comparison_file)
                best.update(progress_row)
                best["comparison_file"] = str(comparison_file)
                result_rows.append(best)
                print(f"Finished OK after {format_seconds(elapsed)}")
                print(f"Comparison file: {comparison_file}")
            else:
                progress_row["status"] = "ok_no_comparison_file_found"
                result_rows.append(progress_row)
                print(f"Finished OK after {format_seconds(elapsed)}, but no comparison CSV found.")

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
            print(f"SETUP ERROR after {format_seconds(elapsed)}: {exc}")

        finally:
            completed_now = len([r for r in progress_rows if r.get("status") in {"ok", "failed", "setup_error", "skipped", "dry_run", "ok_no_comparison_file_found"}])
            elapsed_batch = time.time() - batch_start

            finished_runtimes = [
                r["elapsed_seconds"]
                for r in progress_rows
                if r.get("elapsed_seconds") is not None and pd.notna(r.get("elapsed_seconds"))
            ]

            avg_runtime = sum(finished_runtimes) / len(finished_runtimes) if finished_runtimes else None
            remaining_models = len(model_files) - completed_now
            eta_seconds = avg_runtime * remaining_models if avg_runtime is not None else None

            current_status = {
                "batch_started": datetime.fromtimestamp(batch_start).isoformat(timespec="seconds"),
                "last_updated": datetime.now().isoformat(timespec="seconds"),
                "current_model_index": model_index,
                "total_models": len(model_files),
                "current_model": model_name,
                "completed_models": completed_now,
                "remaining_models": remaining_models,
                "elapsed_batch_seconds": elapsed_batch,
                "elapsed_batch_hms": format_seconds(elapsed_batch),
                "average_finished_model_seconds": avg_runtime,
                "average_finished_model_hms": format_seconds(avg_runtime),
                "estimated_remaining_seconds": eta_seconds,
                "estimated_remaining_hms": format_seconds(eta_seconds),
            }

            write_progress_files(out_root, progress_rows, current_status)

    # Final outputs
    model_info = pd.DataFrame(model_info_rows)
    model_info_path = out_root / "generated_model_info.csv"
    model_info.to_csv(model_info_path, index=False)

    results = pd.DataFrame(result_rows)
    results_path = out_root / "all_model_comparison.csv"

    for col in ["bic", "BIC", "aic", "AIC", "rmse", "RMSE", "rss", "RSS", "cost", "Cost"]:
        if col in results.columns:
            results[col] = pd.to_numeric(results[col], errors="coerce")
            results = results.sort_values(col, ascending=True, na_position="last")
            break

    results.to_csv(results_path, index=False)

    total_elapsed = time.time() - batch_start

    final_status = {
        "batch_started": datetime.fromtimestamp(batch_start).isoformat(timespec="seconds"),
        "batch_finished": datetime.now().isoformat(timespec="seconds"),
        "total_elapsed_seconds": total_elapsed,
        "total_elapsed_hms": format_seconds(total_elapsed),
        "total_models": len(model_files),
        "completed_models": len(progress_rows),
        "model_info": str(model_info_path),
        "comparison": str(results_path),
        "progress_csv": str(out_root / "progress.csv"),
    }

    with (out_root / "status.json").open("w") as f:
        json.dump(final_status, f, indent=2)

    print()
    print("=" * 80)
    print("Batch fitting complete.")
    print("=" * 80)
    print(f"Total runtime:    {format_seconds(total_elapsed)}")
    print(f"Model info:       {model_info_path}")
    print(f"Comparison:       {results_path}")
    print(f"Progress CSV:     {out_root / 'progress.csv'}")
    print(f"Status JSON:      {out_root / 'status.json'}")


if __name__ == "__main__":
    main()
