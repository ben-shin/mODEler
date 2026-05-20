from __future__ import annotations

import json

from odefit.engines.registry import describe_available_engines


def main() -> None:
    descriptions = describe_available_engines()

    print(json.dumps(descriptions, indent=2))


if __name__ == "__main__":
    main()
