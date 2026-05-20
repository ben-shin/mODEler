from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PatchStats:
    file_path: Path
    changed: bool = False
    inserted_signature_params: int = 0
    inserted_call_keywords: int = 0
    inserted_payload_entries: int = 0


SINGLE_SPECIES_FILES = [
    Path("src/odefit/fitting/variable_projection_multistart.py"),
    Path("src/odefit/fitting/variable_projection_model_comparison.py"),
    Path("src/odefit/fitting/variable_projection_multistart_model_comparison.py"),
    Path("src/odefit/fitting/variable_projection_bootstrap.py"),
    Path("src/odefit/fitting/variable_projection_profile_likelihood.py"),
]

MULTISPECIES_FILES = [
    Path("src/odefit/fitting/multispecies_variable_projection_multistart.py"),
    Path("src/odefit/fitting/multispecies_variable_projection_model_comparison.py"),
    Path("src/odefit/fitting/multispecies_variable_projection_multistart_model_comparison.py"),
    Path("src/odefit/fitting/multispecies_variable_projection_bootstrap.py"),
    Path("src/odefit/fitting/multispecies_variable_projection_profile_likelihood.py"),
]

SINGLE_SPECIES_CALLS = [
    "fit_global_observable_model_variable_projection",
    "fit_global_observable_model_variable_projection_multistart",
]

MULTISPECIES_CALLS = [
    "fit_global_observable_model_multispecies_variable_projection",
    "fit_global_observable_model_multispecies_variable_projection_multistart",
]


def _paren_delta(line: str) -> int:
    # Good enough for normal Python call/signature formatting.
    return line.count("(") - line.count(")")


def _leading_whitespace(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def _find_top_level_functions(lines: list[str]):
    """
    Yield dictionaries:
        name
        sig_start
        sig_end
        body_start
        body_end
    """

    i = 0

    while i < len(lines):
        line = lines[i]

        if not line.startswith("def "):
            i += 1
            continue

        name = line.split("def ", 1)[1].split("(", 1)[0].strip()

        sig_start = i
        balance = 0
        sig_end = i

        while sig_end < len(lines):
            balance += _paren_delta(lines[sig_end])

            if balance <= 0 and lines[sig_end].rstrip().endswith(":"):
                break

            sig_end += 1

        body_start = sig_end + 1
        body_end = len(lines)

        j = body_start
        while j < len(lines):
            if lines[j].startswith("def ") or lines[j].startswith("class "):
                body_end = j
                break
            j += 1

        yield {
            "name": name,
            "sig_start": sig_start,
            "sig_end": sig_end,
            "body_start": body_start,
            "body_end": body_end,
        }

        i = body_end


def _function_body_contains_any_call(
    lines: list[str],
    function_info: dict,
    call_names: list[str],
) -> bool:
    body = "".join(
        lines[
            function_info["body_start"] : function_info["body_end"]
        ]
    )

    return any(f"{call_name}(" in body for call_name in call_names)


def _signature_has_engine_name(
    lines: list[str],
    function_info: dict,
) -> bool:
    signature = "".join(
        lines[
            function_info["sig_start"] : function_info["sig_end"] + 1
        ]
    )

    return "engine_name" in signature


def _insert_engine_name_parameter(
    lines: list[str],
    function_info: dict,
) -> None:
    sig_end = function_info["sig_end"]

    if sig_end > function_info["sig_start"]:
        previous_line = lines[sig_end - 1]
        indent = _leading_whitespace(previous_line)

        if not indent:
            indent = "    "
    else:
        indent = "    "

    lines.insert(
        sig_end,
        f'{indent}engine_name: str = "reference",\n',
    )


def _patch_function_signatures(
    lines: list[str],
    call_names: list[str],
) -> int:
    inserted = 0

    # Reverse order so insertion does not invalidate later indices.
    functions = list(_find_top_level_functions(lines))

    for function_info in reversed(functions):
        name = function_info["name"]

        # Worker functions usually receive only a payload dict.
        # Do not mutate worker signatures.
        if "worker" in name:
            continue

        if _signature_has_engine_name(lines, function_info):
            continue

        if not _function_body_contains_any_call(
            lines,
            function_info,
            call_names,
        ):
            continue

        _insert_engine_name_parameter(
            lines,
            function_info,
        )
        inserted += 1

    return inserted


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
        f"Could not find end of call starting at line {start_index + 1}"
    )


def _call_block_has_engine_name(
    lines: list[str],
    start_index: int,
    end_index: int,
) -> bool:
    block = "".join(lines[start_index : end_index + 1])
    return "engine_name=" in block


def _infer_argument_indent(
    lines: list[str],
    start_index: int,
    end_index: int,
) -> str:
    for i in range(start_index + 1, end_index):
        stripped = lines[i].strip()

        if not stripped:
            continue

        if stripped.startswith("#"):
            continue

        if stripped in {")", "),"}:
            continue

        return _leading_whitespace(lines[i])

    return _leading_whitespace(lines[end_index]) + "    "


def _call_appears_inside_payload_worker(
    lines: list[str],
    start_index: int,
    end_index: int,
) -> bool:
    block = "".join(lines[start_index : end_index + 1])

    return (
        "payload[" in block
        or "payload.get(" in block
    )


def _patch_downstream_calls(
    lines: list[str],
    call_names: list[str],
) -> int:
    inserted = 0
    i = 0

    while i < len(lines):
        line = lines[i]

        matched_call_name = None

        for call_name in call_names:
            if f"{call_name}(" in line and not line.lstrip().startswith("def "):
                matched_call_name = call_name
                break

        if matched_call_name is None:
            i += 1
            continue

        end_index = _find_call_end(lines, i)

        if _call_block_has_engine_name(lines, i, end_index):
            i = end_index + 1
            continue

        indent = _infer_argument_indent(lines, i, end_index)

        if _call_appears_inside_payload_worker(lines, i, end_index):
            keyword_line = (
                f'{indent}engine_name=payload.get("engine_name", "reference"),\n'
            )
        else:
            keyword_line = f"{indent}engine_name=engine_name,\n"

        lines.insert(end_index, keyword_line)
        inserted += 1

        i = end_index + 2

    return inserted


def _patch_bootstrap_payload_entries(lines: list[str]) -> int:
    """
    Add "engine_name": engine_name after "method": method in payload dicts.

    This is deliberately conservative and idempotent.
    """

    inserted = 0
    i = 0

    while i < len(lines):
        line = lines[i]

        if '"method": method,' not in line:
            i += 1
            continue

        lookahead = "".join(lines[i : min(i + 8, len(lines))])

        if '"engine_name"' in lookahead:
            i += 1
            continue

        indent = _leading_whitespace(line)

        lines.insert(
            i + 1,
            f'{indent}"engine_name": engine_name,\n',
        )

        inserted += 1
        i += 2

    return inserted


def patch_file(
    file_path: Path,
    call_names: list[str],
    *,
    dry_run: bool,
    backup: bool,
) -> PatchStats:
    stats = PatchStats(file_path=file_path)

    if not file_path.exists():
        print(f"SKIP missing file: {file_path}")
        return stats

    original_text = file_path.read_text()
    lines = original_text.splitlines(keepends=True)

    stats.inserted_signature_params = _patch_function_signatures(
        lines,
        call_names,
    )

    stats.inserted_call_keywords = _patch_downstream_calls(
        lines,
        call_names,
    )

    if "bootstrap" in file_path.name:
        stats.inserted_payload_entries = _patch_bootstrap_payload_entries(
            lines,
        )

    new_text = "".join(lines)

    stats.changed = new_text != original_text

    if stats.changed and not dry_run:
        if backup:
            backup_path = file_path.with_suffix(file_path.suffix + ".bak_engine_name")
            backup_path.write_text(original_text)

        file_path.write_text(new_text)

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Propagate engine_name through variable projection wrapper files."
        )
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show intended changes without writing files.",
    )

    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not write .bak_engine_name backup files.",
    )

    args = parser.parse_args()

    all_stats: list[PatchStats] = []

    for file_path in SINGLE_SPECIES_FILES:
        all_stats.append(
            patch_file(
                file_path,
                SINGLE_SPECIES_CALLS,
                dry_run=args.dry_run,
                backup=not args.no_backup,
            )
        )

    for file_path in MULTISPECIES_FILES:
        all_stats.append(
            patch_file(
                file_path,
                MULTISPECIES_CALLS,
                dry_run=args.dry_run,
                backup=not args.no_backup,
            )
        )

    print("\nEngine-name propagation summary")
    print("===============================")

    for stats in all_stats:
        status = "CHANGED" if stats.changed else "unchanged"

        print(
            f"{status}: {stats.file_path} "
            f"signatures={stats.inserted_signature_params} "
            f"calls={stats.inserted_call_keywords} "
            f"payloads={stats.inserted_payload_entries}"
        )

    changed_count = sum(stats.changed for stats in all_stats)

    print(f"\nChanged files: {changed_count}")

    if args.dry_run:
        print("\nDry run only. No files were written.")


if __name__ == "__main__":
    main()
