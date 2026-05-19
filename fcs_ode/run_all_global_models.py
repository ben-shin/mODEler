#!/usr/bin/env python3

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd


# ============================================================
# Model parsing
# ============================================================

def strip_comment(line: str) -> str:
    return line.split("#", 1)[0].strip()


def parse_side(side: str):
    """
    Parse a reaction side into {species: stoichiometric coefficient}.

    Examples:
      2P1        -> {"P1": 2}
      P1+P2      -> {"P1": 1, "P2": 1}
      2P1 + P2   -> {"P1": 2, "P2": 1}
    """
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
    """
    Infer reactions, species, and mODEler parameter names from a simple model txt.

    Assumption:
      line 1 irreversible A>B gives k1f
      line 1 reversible A-B gives k1f and k1r
      line 2 gives k2f/k2r, etc.

    This matches the usual compact mODEler reaction syntax.
    """
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
    """
    For species named P1, P2, P12, return 1, 2, 12.
    Otherwise return None.
    """
    match = re.fullmatch(r"P(\d+)", species)
    if match is None:
        return None
    return int(match.group(1))


def check_mass_balance(parsed_reactions):
    """
    Checks mass conservation for Pn-style species.

    Example:
      2P1-P2 is balanced because left mass = 2*1 and right mass = 1*2.
      P1+P2-P3 is balanced because left mass = 1+2 and right mass = 3.

    If non-Pn species are present, returns unchecked.
    """
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
    """
    Build a mODEler global observable fitting config.

    Mass conservation is not a separate switch here.
    It is enforced by stoichiometrically balanced reactions.
    This config sets P1(0)=100 uM and all other species to zero.
    """
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
        if s == initial_species:
            value = initial_conc
        else:
            value = 0.0

        config["initial_conditions"][s] = {
            "value": value,
            "mode": "fixed",
            "lower_bound": 0.0,
            "upper_bound": initial_conc,
        }

    return config


# ============================================================
# Results collection
# ============================================================

def find_comparison_files(output_dir: Path):
    """
    Find likely mODEler comparison CSVs.
    """
    if not output_dir.exists():
        return []

    candidates = []

    preferred_names = [
        "parallel_multistart_comparison.csv",
        "multistart_comparison.csv",
        "comparison.csv",
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


def best_row_from_comparison(comparison_csv: Path):
    """
    Return best row from a comparison CSV, sorted by BIC/AIC/RMSE/RSS/cost if available.
    """
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


def stream_subprocess_to_log(cmd, log_path: Path):
    """
    Run subprocess while printing output to terminal and saving it to a log file.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w") as log:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert process.stdout is not None

        for line in process.stdout:
            print(line, end="")
            log.write(line)
            log.flush()

        return process.wait()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate mODEler global-observable configs and fit all .txt model files "
            "to an FCS timecourse with P1(0)=100 uM and mass-conserving stoichiometry."
        )
    )

    parser.add_argument(
        "--model-dir",
        default="fcs_ode",
        help="Directory containing model .txt files. Default: fcs_ode",
    )

    parser.add_argument(
        "--data",
        default="fcs_ode/modeler_by_time_clean.csv",
        help="Input CSV. Default: fcs_ode/modeler_by_time_clean.csv",
    )

    parser.add_argument(
        "--config-dir",
        default="fcs_ode/configs_P1_100uM",
        help="Directory for generated config JSON files.",
    )

    parser.add_argument(
        "--out-root",
        default="fcs_ode/fits_P1_100uM",
        help="Root output directory for all model fits.",
    )

    parser.add_argument(
        "--time-column",
        default=None,
        help="Time column name. Default: auto-detect time_min, time, or first column.",
    )

    parser.add_argument(
        "--initial-species",
        default="P1",
        help="Species with nonzero initial concentration. Default: P1",
    )

    parser.add_argument(
        "--initial-conc",
        type=float,
        default=100.0,
        help="Initial concentration of initial species. Default: 100.0",
    )

    parser.add_argument(
        "--observed-species",
        default="P1",
        help=(
            "Species used as the shared ODE observable. "
            "Default: P1, so each FCS column is fitted as scale_i*P1(t)+offset_i."
        ),
    )

    parser.add_argument(
        "--n-starts",
        type=int,
        default=50,
        help="Number of multistart fits per model. Default: 50",
    )

    parser.add_argument(
        "--n-workers",
        type=int,
        default=8,
        help="Parallel workers inside each mODEler multistart run. Default: 8",
    )

    parser.add_argument(
        "--rate-guess",
        type=float,
        default=1e-7,
        help="Initial guess for each kinetic parameter. Default: 1e-7",
    )

    parser.add_argument(
        "--rate-lower",
        type=float,
        default=1e-15,
        help="Lower bound for kinetic parameters. Default: 1e-15",
    )

    parser.add_argument(
        "--rate-upper",
        type=float,
        default=1e-1,
        help="Upper bound for kinetic parameters. Default: 1e-1",
    )

    parser.add_argument(
        "--max-nfev",
        type=int,
        default=10000,
        help="Max function evaluations per optimizer run. Default: 10000",
    )

    parser.add_argument(
        "--plots",
        action="store_true",
        help="Enable mODEler plots. Default is no plots.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate configs and model info but do not run fitting.",
    )

    parser.add_argument(
        "--skip-missing-species",
        action="store_true",
        help=(
            "Skip models that do not contain the initial/observed species "
            "instead of treating them as setup errors."
        ),
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

    # Detect time column.
    df_head = pd.read_csv(data_path, nrows=5)

    if args.time_column is not None:
        time_column = args.time_column
    elif "time_min" in df_head.columns:
        time_column = "time_min"
    elif "time" in df_head.columns:
        time_column = "time"
    else:
        time_column = df_head.columns[0]

    if time_column not in df_head.columns:
        raise ValueError(
            f"Time column {time_column!r} not found in {data_path}. "
            f"Available columns: {list(df_head.columns)}"
        )

    model_files = sorted(
        p for p in model_dir.glob("*.txt")
        if p.name != "requirements.txt"
    )

    if not model_files:
        raise RuntimeError(f"No model .txt files found in {model_dir}")

    print("=" * 80)
    print("mODEler global observable batch fitting")
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
    print(f"Dry run:                {args.dry_run}")
    print("=" * 80)
    print()

    model_info_rows = []
    result_rows = []

    for model_index, model_path in enumerate(model_files, start=1):
        model_name = model_path.stem
        output_dir = out_root / model_name
        config_path = config_dir / f"{model_name}_global_multistart.json"
        log_path = output_dir / "run.log"

        print()
        print("=" * 80)
        print(f"[{model_index}/{len(model_files)}] Model: {model_name}")
        print("=" * 80)
        print(f"Model file: {model_path}")

        try:
            reactions, parsed_reactions, species, parameters = parse_model_file(model_path)
            mass_warnings = check_mass_balance(parsed_reactions)

            if args.initial_species not in species or args.observed_species not in species:
                msg = (
                    f"Required species missing. "
                    f"initial_species={args.initial_species!r}, "
                    f"observed_species={args.observed_species!r}, "
                    f"detected species={species}"
                )

                if args.skip_missing_species:
                    print(f"SKIPPING: {msg}")

                    model_info_rows.append({
                        "model": model_name,
                        "model_file": str(model_path),
                        "status": "skipped_missing_species",
                        "message": msg,
                        "species": ";".join(species),
                        "parameters": ";".join(parameters),
                        "n_species": len(species),
                        "n_parameters": len(parameters),
                        "n_reactions": len(reactions),
                        "mass_balance_warnings": " | ".join(mass_warnings),
                    })

                    result_rows.append({
                        "model": model_name,
                        "status": "skipped_missing_species",
                        "message": msg,
                    })

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

            print(f"Species:       {', '.join(species)}")
            print(f"Parameters:    {', '.join(parameters)}")
            print(f"Reactions:     {len(reactions)}")
            print(f"Config:        {config_path}")
            print(f"Output dir:    {output_dir}")

            if mass_warnings:
                print("Mass-balance warnings:")
                for w in mass_warnings:
                    print(f"  - {w}")
            else:
                print("Mass-balance check: OK for Pn-style reactions.")

            model_info_rows.append({
                "model": model_name,
                "model_file": str(model_path),
                "config_file": str(config_path),
                "output_dir": str(output_dir),
                "status": "config_generated",
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

            if args.dry_run:
                print("Dry run: not fitting this model.")
                result_rows.append({
                    "model": model_name,
                    "status": "dry_run_config_generated",
                    "output_dir": str(output_dir),
                    "config_file": str(config_path),
                })
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

            print()
            print("Running command:")
            print(" ".join(cmd))
            print()

            return_code = stream_subprocess_to_log(cmd, log_path)

            if return_code != 0:
                print(f"FAILED: {model_name} returned code {return_code}")
                result_rows.append({
                    "model": model_name,
                    "status": "failed",
                    "return_code": return_code,
                    "output_dir": str(output_dir),
                    "config_file": str(config_path),
                    "log_file": str(log_path),
                })
                continue

            comparison_files = find_comparison_files(output_dir)

            if comparison_files:
                comparison_file = comparison_files[0]
                best = best_row_from_comparison(comparison_file)
                best["model"] = model_name
                best["status"] = "ok"
                best["return_code"] = return_code
                best["output_dir"] = str(output_dir)
                best["config_file"] = str(config_path)
                best["log_file"] = str(log_path)
                best["n_species"] = len(species)
                best["n_parameters"] = len(parameters)
                best["n_reactions"] = len(reactions)
                result_rows.append(best)

                print()
                print(f"Finished OK. Comparison file: {comparison_file}")
            else:
                result_rows.append({
                    "model": model_name,
                    "status": "ok_no_comparison_file_found",
                    "return_code": return_code,
                    "output_dir": str(output_dir),
                    "config_file": str(config_path),
                    "log_file": str(log_path),
                    "n_species": len(species),
                    "n_parameters": len(parameters),
                    "n_reactions": len(reactions),
                })

                print()
                print("Finished OK, but no comparison CSV found.")

        except Exception as exc:
            print(f"SETUP ERROR for {model_name}: {exc}")

            result_rows.append({
                "model": model_name,
                "status": "setup_error",
                "error": str(exc),
                "model_file": str(model_path),
            })

    # Write summary files.
    model_info = pd.DataFrame(model_info_rows)
    model_info_path = out_root / "generated_model_info.csv"
    model_info.to_csv(model_info_path, index=False)

    results = pd.DataFrame(result_rows)
    results_path = out_root / "all_model_comparison.csv"

    # Sort model comparison by common fit metrics if present.
    sort_candidates = ["bic", "BIC", "aic", "AIC", "rmse", "RMSE", "rss", "RSS", "cost", "Cost"]

    for col in sort_candidates:
        if col in results.columns:
            results[col] = pd.to_numeric(results[col], errors="coerce")
            results = results.sort_values(col, ascending=True, na_position="last")
            break

    results.to_csv(results_path, index=False)

    print()
    print("=" * 80)
    print("Batch fitting complete.")
    print("=" * 80)
    print(f"Wrote model info:  {model_info_path}")
    print(f"Wrote comparison:  {results_path}")
    print()
    print("To inspect the ranked results:")
    print(f"  python - <<'PY'\nimport pandas as pd\ndf = pd.read_csv('{results_path}')\nprint(df.head(20).to_string(index=False))\nPY")


if __name__ == "__main__":
    main()
