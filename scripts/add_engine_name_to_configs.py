from __future__ import annotations

import argparse
import json
from pathlib import Path


CONFIG_PATHS = [
    "examples/configs/global_hsqc_variable_projection_config.json",
    "examples/configs/global_hsqc_variable_projection_multistart_config.json",
    "examples/configs/global_hsqc_variable_projection_model_comparison_config.json",
    "examples/configs/global_hsqc_variable_projection_multistart_model_comparison_config.json",
    "examples/configs/global_hsqc_variable_projection_bootstrap_config.json",
    "examples/configs/global_hsqc_variable_projection_profile_likelihood_config.json",
    "examples/configs/global_hsqc_multispecies_variable_projection_config.json",
    "examples/configs/global_hsqc_multispecies_variable_projection_multistart_config.json",
    "examples/configs/global_hsqc_multispecies_variable_projection_model_comparison_config.json",
    "examples/configs/global_hsqc_multispecies_variable_projection_multistart_model_comparison_config.json",
    "examples/configs/global_hsqc_multispecies_variable_projection_bootstrap_config.json",
    "examples/configs/global_hsqc_multispecies_variable_projection_profile_likelihood_config.json",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Add "engine_name": "reference" to example configs.'
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show changes without writing files.",
    )

    args = parser.parse_args()

    changed = 0
    skipped = 0
    missing = 0

    for raw_path in CONFIG_PATHS:
        path = Path(raw_path)

        if not path.exists():
            print(f"MISSING: {path}")
            missing += 1
            continue

        data = json.loads(path.read_text())

        if data.get("engine_name") == "reference":
            print(f"unchanged: {path}")
            skipped += 1
            continue

        data["engine_name"] = "reference"

        print(f"CHANGED: {path}")
        changed += 1

        if not args.dry_run:
            path.write_text(json.dumps(data, indent=2) + "\n")

    print("\nConfig engine-name summary")
    print("==========================")
    print(f"Changed: {changed}")
    print(f"Unchanged: {skipped}")
    print(f"Missing: {missing}")

    if args.dry_run:
        print("\nDry run only. No files were written.")


if __name__ == "__main__":
    main()
