from __future__ import annotations

import argparse
import json
from pathlib import Path

from odefit.api.project_config import (
    collect_engine_names,
    load_project_payload,
    save_project_payload,
    validate_project_engines,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Round-trip a project/config JSON and report engine_name persistence."
    )

    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    payload = load_project_payload(input_path)
    save_project_payload(payload, output_path)

    reloaded = load_project_payload(output_path)

    engine_names = collect_engine_names(reloaded)
    validation = validate_project_engines(reloaded)

    print("\nProject/config round-trip complete")
    print("==================================")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")

    print("\nEngine names:")
    for path, engine_name in engine_names.items():
        print(f"  {path}: {engine_name}")

    print("\nValidation:")
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
