from __future__ import annotations

import argparse
import re
from pathlib import Path


CLI_PATH = Path("src/odefit/cli.py")

ENGINE_READ_LINE = '    engine_name = str(config.get("engine_name", "reference"))\n'


ENGINE_AWARE_CALLS = [
    "fit_global_observable_model_variable_projection(",
    "fit_global_observable_model_multispecies_variable_projection(",
    "fit_global_observable_variable_projection_multistart(",
    "fit_global_observable_model_multispecies_variable_projection_multistart(",
    "fit_global_observable_variable_projection_model_comparison(",
    "fit_global_observable_multispecies_variable_projection_model_comparison(",
    "fit_global_observable_variable_projection_multistart_model_comparison(",
    "fit_global_observable_multispecies_variable_projection_multistart_model_comparison(",
    "bootstrap_global_observable_variable_projection_fit(",
    "bootstrap_global_observable_multispecies_variable_projection_fit(",
    "fit_variable_projection_profile_likelihood(",
    "fit_multispecies_variable_projection_profile_likelihood(",
]


def _paren_delta(line: str) -> int:
    return line.count("(") - line.count(")")


def _find_signature_end(lines: list[str], def_index: int) -> int:
    balance = 0

    for index in range(def_index, len(lines)):
        balance += _paren_delta(lines[index])

        if balance <= 0 and lines[index].rstrip().endswith(":"):
            return index

    raise RuntimeError(f"Could not find signature end after line {def_index + 1}")


def _find_function_end(lines: list[str], def_index: int) -> int:
    signature_end = _find_signature_end(lines, def_index)

    for index in range(signature_end + 1, len(lines)):
        if lines[index].startswith("def ") or lines[index].startswith("class "):
            return index

    return len(lines)


def _remove_bad_engine_lines_from_signatures(lines: list[str]) -> int:
    removed = 0
    index = 0

    while index < len(lines):
        line = lines[index]

        if not line.startswith("def command_"):
            index += 1
            continue

        signature_end = _find_signature_end(lines, index)

        scan_index = index + 1

        while scan_index <= signature_end:
            if "engine_name = str(config.get" in lines[scan_index]:
                del lines[scan_index]
                removed += 1
                signature_end -= 1
                continue

            scan_index += 1

        index = signature_end + 1

    return removed


def _function_body(lines: list[str], def_index: int, function_end: int) -> str:
    signature_end = _find_signature_end(lines, def_index)
    return "".join(lines[signature_end + 1 : function_end])


def _function_uses_engine_workflow(body: str) -> bool:
    return any(call in body for call in ENGINE_AWARE_CALLS)


def _function_has_config_arg(lines: list[str], def_index: int) -> bool:
    signature_end = _find_signature_end(lines, def_index)
    signature = "".join(lines[def_index : signature_end + 1])
    return "config:" in signature or "config: dict" in signature


def _function_already_has_engine_read(body: str) -> bool:
    return 'engine_name = str(config.get("engine_name", "reference"))' in body


def _find_docstring_end_or_body_start(lines: list[str], def_index: int) -> int:
    signature_end = _find_signature_end(lines, def_index)

    index = signature_end + 1

    while index < len(lines) and not lines[index].strip():
        index += 1

    if index < len(lines) and lines[index].lstrip().startswith('"""'):
        index += 1

        while index < len(lines) and '"""' not in lines[index]:
            index += 1

        if index < len(lines):
            index += 1

    while index < len(lines) and not lines[index].strip():
        index += 1

    return index


def _insert_engine_reads(lines: list[str]) -> int:
    inserted = 0

    def_indices = [
        index
        for index, line in enumerate(lines)
        if line.startswith("def command_")
    ]

    for def_index in reversed(def_indices):
        function_end = _find_function_end(lines, def_index)
        body = _function_body(lines, def_index, function_end)

        if not _function_uses_engine_workflow(body):
            continue

        if _function_already_has_engine_read(body):
            continue

        if _function_has_config_arg(lines, def_index):
            insert_at = _find_docstring_end_or_body_start(lines, def_index)
            lines.insert(insert_at, ENGINE_READ_LINE)
            inserted += 1
            continue

        # Functions without config argument usually load config internally.
        for index in range(def_index, function_end):
            if "config = load_fit_config(args.config)" in lines[index]:
                lines.insert(index + 1, ENGINE_READ_LINE)
                inserted += 1
                break

    return inserted


def _find_call_end(lines: list[str], start_index: int) -> int:
    balance = 0
    started = False

    for index in range(start_index, len(lines)):
        balance += _paren_delta(lines[index])

        if "(" in lines[index]:
            started = True

        if started and balance <= 0:
            return index

    raise RuntimeError(f"Could not find call end after line {start_index + 1}")


def _infer_argument_indent(lines: list[str], start_index: int, end_index: int) -> str:
    for index in range(start_index + 1, end_index):
        stripped = lines[index].strip()

        if not stripped or stripped.startswith("#"):
            continue

        if stripped in {")", "),"}:
            continue

        return lines[index][: len(lines[index]) - len(lines[index].lstrip())]

    return "        "


def _insert_engine_name_into_calls(lines: list[str]) -> int:
    inserted = 0
    index = 0

    while index < len(lines):
        line = lines[index]

        if line.lstrip().startswith("def "):
            index += 1
            continue

        if not any(call in line for call in ENGINE_AWARE_CALLS):
            index += 1
            continue

        call_end = _find_call_end(lines, index)
        block = "".join(lines[index : call_end + 1])

        if "engine_name=" in block:
            index = call_end + 1
            continue

        indent = _infer_argument_indent(lines, index, call_end)
        lines.insert(call_end, f"{indent}engine_name=engine_name,\n")
        inserted += 1

        index = call_end + 2

    return inserted


def _remove_duplicate_engine_reads(lines: list[str]) -> int:
    removed = 0
    index = 0

    while index < len(lines):
        if not lines[index].startswith("def command_"):
            index += 1
            continue

        function_end = _find_function_end(lines, index)
        seen = False
        scan_index = index + 1

        while scan_index < function_end:
            if lines[scan_index] == ENGINE_READ_LINE:
                if seen:
                    del lines[scan_index]
                    removed += 1
                    function_end -= 1
                    continue

                seen = True

            scan_index += 1

        index = function_end

    return removed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Repair cli.py engine_name propagation safely."
    )

    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-backup", action="store_true")

    args = parser.parse_args()

    original = CLI_PATH.read_text()
    lines = original.splitlines(keepends=True)

    removed_bad_signature_lines = _remove_bad_engine_lines_from_signatures(lines)
    inserted_reads = _insert_engine_reads(lines)
    inserted_call_keywords = _insert_engine_name_into_calls(lines)
    removed_duplicate_reads = _remove_duplicate_engine_reads(lines)

    new = "".join(lines)
    changed = new != original

    print("\nCLI engine_name repair summary")
    print("==============================")
    print(f"Changed: {changed}")
    print(f"Removed bad signature lines: {removed_bad_signature_lines}")
    print(f"Inserted engine reads: {inserted_reads}")
    print(f"Inserted call keywords: {inserted_call_keywords}")
    print(f"Removed duplicate engine reads: {removed_duplicate_reads}")

    if args.dry_run:
        print("\nDry run only. No files written.")
        return

    if changed:
        if not args.no_backup:
            CLI_PATH.with_suffix(CLI_PATH.suffix + ".bak_before_engine_repair").write_text(
                original
            )

        CLI_PATH.write_text(new)


if __name__ == "__main__":
    main()
