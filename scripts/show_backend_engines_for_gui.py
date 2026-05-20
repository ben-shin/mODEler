from __future__ import annotations

import argparse
import json
from pathlib import Path

from odefit.api.backend import get_backend_engine_capabilities


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print backend engine capabilities as GUI-friendly JSON."
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Optional output JSON file.",
    )

    args = parser.parse_args()

    payload = get_backend_engine_capabilities()

    text = json.dumps(payload, indent=2)

    print(text)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n")


if __name__ == "__main__":
    main()
