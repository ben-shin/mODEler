from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


API_FILE = Path("src/odefit/api/backend.py")
CLI_FILE = Path("src/odefit/cli.py")


WORKFLOW_CALLS = [
    "fit_global_observable_model_variable_projection",
    "fit_global_observable_model_multispecies_variable_projection",
    "fit_global_observable_model_variable_projection_multistart",
    "fit_global_observable_model_multispecies_variable_projection_multistart",
    "fit_global_observable_variable_projection_model_comparison",
    "fit_global_observable_multispecies_variable_projection_model_comparison",
    "fit_global_observable_variable_projection_multistart_model_comparison",
    "fit_global_observable_multispecies_variable_projection_multistart_model_comparison",
    "bootstrap_global_observable_variable_projection_fit",
    "bootstrap_global_observable_multispecies_variable_projection_fit",
    "fit_variable_projection_profile_likelihood",
    "fit_multispecies_variable_projection_profile_likelihood",
]


@dataclass
class PatchResult:
    file_path: Path
    changed: bool
    inserted_helpers: int = 0
    inserted_engine_reads: int = 0
    inserted_call_keywords: int = 0


def _paren_delta(line: str) -> int:
    return line.count("(") - line.count(")")


def _leading_whitespace(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def _find_call_end(lines: list[str], start_index: int) -> int:
    balance = 0
    started = False

    for i in range(start_index, len(lines)):
        line = lines[i]
        balance += _paren_delta(line)

        if "(" in line:
            started = True

        if started and balance <= 0:
            return i

    raise RuntimeError(
        f"Could not find call end starting at line {start_index + 1}"
    )


def _call_block_has_keyword(
    lines: list[str],
    start_index: int,
    end_index: int,
    keyword: str,
) -> bool:
    block = "".join(lines[start_index : end_index + 1])
    return f"{keyword}=" in block


def _infer_argument_indent(
    lines: list[str],
    start_index: int,
    end_index: int,
) -> str:
    for i in range(start_index + 1, end_index):
        stripped = lines[i].strip()

        if not stripped or stripped.startswith("#"):
            continue

        if stripped in {")", "),"}:
            continue

        return _leading_whitespace(lines[i])

    return _leading_whitespace(lines[end_index]) + "    "


def _patch_call_keywords(lines: list[str]) -> int:
    inserted = 0
    i = 0

    while i < len(lines):
        line = lines[i]

        matched = None

        for call_name in WORKFLOW_CALLS:
            if f"{call_name}(" in line and not line.lstrip().startswith("def "):
                matched = call_name
                break

        if matched is None:
            i += 1
            continue

        end_index = _find_call_end(lines, i)

        if _call_block_has_keyword(lines, i, end_index, "engine_name"):
            i = end_index + 1
            continue

        indent = _infer_argument_indent(lines, i, end_index)
        lines.insert(end_index, f"{indent}engine_name=engine_name,\n")
        inserted += 1

        i = end_index + 2

    return inserted


def _insert_api_helper(lines: list[str]) -> int:
    text = "".join(lines)

    if "def _engine_name_from_config(" in text:
        return 0

    insert_after = None

    for i, line in enumerate(lines):
        if line.startswith("def _load_config("):
            end = _find_function_end(lines, i)
            insert_after = end
            break

    if insert_after is None:
        raise RuntimeError("Could not find _load_config in API backend.")

    helper = [
        "\n",
        "\n",
        "def _engine_name_from_config(config: dict) -> str:\n",
        '    return str(config.get("engine_name", "reference"))\n',
    ]

    lines[insert_after:insert_after] = helper

    return 1


def _find_function_end(lines: list[str], def_index: int) -> int:
    i = def_index + 1

    while i < len(lines):
        if lines[i].startswith("def ") or lines[i].startswith("class "):
            return i
        i += 1

    return len(lines)


def _patch_api_engine_reads(lines: list[str]) -> int:
    inserted = 0

    target_functions = {
        "fit_global_observables_from_config",
        "compare_global_observables_from_config",
        "bootstrap_global_observables_from_config",
        "profile_likelihood_global_observables_from_config",
    }

    i = 0

    while i < len(lines):
        line = lines[i]

        if not line.startswith("def "):
            i += 1
            continue

        function_name = line.split("def ", 1)[1].split("(", 1)[0]

        if function_name not in target_functions:
            i += 1
            continue

        end = _find_function_end(lines, i)
        body = "".join(lines[i:end])

        if "engine_name = _engine_name_from_config(config)" in body:
            i = end
            continue

        insert_at = None

        for j in range(i, end):
            if "config = _load_config(" in lines[j]:
                insert_at = j + 1
                break

        if insert_at is None:
            i = end
            continue

        lines.insert(
            insert_at,
            "    engine_name = _engine_name_from_config(config)\n",
        )
        inserted += 1
        i = end + 1

    return inserted


def _patch_cli_engine_reads(lines: list[str]) -> int:
    """
    Add:
        engine_name = str(config.get("engine_name", "reference"))

    after config load or near top of config-based command functions,
    but only if the function calls an engine-aware workflow.
    """

    inserted = 0
    i = 0

    while i < len(lines):
        line = lines[i]

        if not line.startswith("def command_"):
            i += 1
            continue

        function_name = line.split("def ", 1)[1].split("(", 1)[0]
        end = _find_function_end(lines, i)
        body = "".join(lines[i:end])

        if not any(f"{call_name}(" in body for call_name in WORKFLOW_CALLS):
            i = end
            continue

        if 'engine_name = str(config.get("engine_name", "reference"))' in body:
            i = end
            continue

        insert_at = None

        for j in range(i, end):
            if "config = load_fit_config(" in lines[j]:
                insert_at = j + 1
                break

        # Some helper command functions receive config as an argument and do not call load_fit_config.
        # In those, insert immediately after the docstring/simple first lines if not already present.
        if insert_at is None:
            for j in range(i + 1, min(i + 15, end)):
                if lines[j].strip() and not lines[j].strip().startswith('"""'):
                    insert_at = j + 1
                    break

        if insert_at is None:
            i = end
            continue

        lines.insert(
            insert_at,
            '    engine_name = str(config.get("engine_name", "reference"))\n',
        )
        inserted += 1
        i = end + 1

    return inserted


def patch_api_file(
    *,
    dry_run: bool,
    backup: bool,
) -> PatchResult:
    original = API_FILE.read_text()
    lines = original.splitlines(keepends=True)

    inserted_helpers = _insert_api_helper(lines)
    inserted_reads = _patch_api_engine_reads(lines)
    inserted_keywords = _patch_call_keywords(lines)

    new = "".join(lines)
    changed = new != original

    if changed and not dry_run:
        if backup:
            API_FILE.with_suffix(API_FILE.suffix + ".bak_engine_name").write_text(
                original
            )
        API_FILE.write_text(new)

    return PatchResult(
        file_path=API_FILE,
        changed=changed,
        inserted_helpers=inserted_helpers,
        inserted_engine_reads=inserted_reads,
        inserted_call_keywords=inserted_keywords,
    )


def patch_cli_file(
    *,
    dry_run: bool,
    backup: bool,
) -> PatchResult:
    original = CLI_FILE.read_text()
    lines = original.splitlines(keepends=True)

    inserted_reads = _patch_cli_engine_reads(lines)
    inserted_keywords = _patch_call_keywords(lines)

    new = "".join(lines)
    changed = new != original

    if changed and not dry_run:
        if backup:
            CLI_FILE.with_suffix(CLI_FILE.suffix + ".bak_engine_name").write_text(
                original
            )
        CLI_FILE.write_text(new)

    return PatchResult(
        file_path=CLI_FILE,
        changed=changed,
        inserted_helpers=0,
        inserted_engine_reads=inserted_reads,
        inserted_call_keywords=inserted_keywords,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Propagate engine_name through API backend and CLI calls."
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show changes without writing files.",
    )

    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not write backup files.",
    )

    args = parser.parse_args()

    results = [
        patch_api_file(
            dry_run=args.dry_run,
            backup=not args.no_backup,
        ),
        patch_cli_file(
            dry_run=args.dry_run,
            backup=not args.no_backup,
        ),
    ]

    print("\nAPI/CLI engine-name propagation summary")
    print("=======================================")

    for result in results:
        status = "CHANGED" if result.changed else "unchanged"

        print(
            f"{status}: {result.file_path} "
            f"helpers={result.inserted_helpers} "
            f"engine_reads={result.inserted_engine_reads} "
            f"call_keywords={result.inserted_call_keywords}"
        )

    if args.dry_run:
        print("\nDry run only. No files were written.")


if __name__ == "__main__":
    main()
