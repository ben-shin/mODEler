from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


@dataclass
class TriageFinding:
    category: str
    severity: str
    title: str
    evidence: str
    recommendation: str


def _safe_float(value, default=None):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _load_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    dataframe = pd.read_csv(path)

    for column in [
        "median_seconds",
        "mean_seconds",
        "speedup_vs_reference_median",
        "speedup_vs_reference_mean",
        "n_timepoints",
        "n_observables",
        "n_repeats",
        "mean_nfev",
        "median_k1f",
        "median_rss",
        "rss",
    ]:
        if column in dataframe.columns:
            dataframe[column] = pd.to_numeric(
                dataframe[column],
                errors="coerce",
            )

    return dataframe


def _best_speedup_for_benchmark(
    dataframe: pd.DataFrame,
    benchmark_contains: str,
) -> float | None:
    if dataframe.empty:
        return None

    subset = dataframe[
        dataframe["benchmark"].astype(str).str.contains(
            benchmark_contains,
            case=False,
            na=False,
        )
        & dataframe["available"].astype(bool)
        & (dataframe["engine_name"] != "reference")
    ]

    if subset.empty:
        return None

    values = subset["speedup_vs_reference_median"].dropna()

    if values.empty:
        return None

    return float(values.max())


def _available_engines(dataframe: pd.DataFrame) -> list[str]:
    if dataframe.empty or "available" not in dataframe.columns:
        return []

    subset = dataframe[dataframe["available"].astype(bool)]

    return sorted(
        str(engine)
        for engine in subset["engine_name"].dropna().unique()
    )


def _unavailable_engines(dataframe: pd.DataFrame) -> list[str]:
    if dataframe.empty or "available" not in dataframe.columns:
        return []

    subset = dataframe[~dataframe["available"].astype(bool)]

    return sorted(
        str(engine)
        for engine in subset["engine_name"].dropna().unique()
    )


def triage_benchmarks(dataframe: pd.DataFrame) -> list[TriageFinding]:
    findings: list[TriageFinding] = []

    if dataframe.empty:
        return [
            TriageFinding(
                category="benchmark_coverage",
                severity="high",
                title="No benchmark summary data found",
                evidence="The benchmark summary CSV is empty or missing.",
                recommendation=(
                    "Run the benchmark sequence first, then rerun triage: "
                    "projection kernel, batched projection, production batch "
                    "methods, variable-projection workflow, then summary."
                ),
            )
        ]

    available = _available_engines(dataframe)
    unavailable = _unavailable_engines(dataframe)

    findings.append(
        TriageFinding(
            category="engine_coverage",
            severity="info",
            title="Available benchmarked engines",
            evidence=", ".join(available) if available else "No available engines reported.",
            recommendation=(
                "Keep reference as the baseline. Treat optional engines as "
                "experimental until workflow-level benchmarks show stable gains."
            ),
        )
    )

    if unavailable:
        findings.append(
            TriageFinding(
                category="engine_coverage",
                severity="medium",
                title="Some engines were unavailable",
                evidence=", ".join(unavailable),
                recommendation=(
                    "This is acceptable for optional engines. If GPU/JAX work is "
                    "planned, install CUDA-enabled jaxlib and rerun the benchmark suite."
                ),
            )
        )

    kernel_speedup = _best_speedup_for_benchmark(
        dataframe,
        "single_species_projection",
    )

    batch_speedup = _best_speedup_for_benchmark(
        dataframe,
        "project_single_species_batch",
    )

    workflow_speedup = _best_speedup_for_benchmark(
        dataframe,
        "single_species_variable_projection_fit",
    )

    if kernel_speedup is not None:
        severity = "info" if kernel_speedup >= 1.2 else "low"

        findings.append(
            TriageFinding(
                category="projection_kernel",
                severity=severity,
                title="Projection kernel speedup measured",
                evidence=f"Best non-reference projection-kernel speedup: {kernel_speedup:.3g}x.",
                recommendation=(
                    "Kernel speedup is useful, but do not overinterpret it. "
                    "Projection may be only a small fraction of full fitting time."
                ),
            )
        )

    if batch_speedup is not None:
        if batch_speedup >= 1.2:
            severity = "medium"
            recommendation = (
                "Batched projection appears worth productionizing further. "
                "Next targets: reduce residual DataFrame construction and avoid "
                "per-evaluation pandas overhead in variable projection."
            )
        else:
            severity = "low"
            recommendation = (
                "Batched projection is not clearly faster than reference yet. "
                "Investigate memory transfer overhead, JIT warmup, and whether "
                "the reference NumPy vectorized path is already near optimal."
            )

        findings.append(
            TriageFinding(
                category="batched_projection",
                severity=severity,
                title="Production batch projection speedup measured",
                evidence=f"Best non-reference batch-method speedup: {batch_speedup:.3g}x.",
                recommendation=recommendation,
            )
        )

    if workflow_speedup is not None:
        if workflow_speedup >= 1.2:
            findings.append(
                TriageFinding(
                    category="workflow",
                    severity="high",
                    title="Engine acceleration improves full workflow",
                    evidence=f"Best non-reference workflow speedup: {workflow_speedup:.3g}x.",
                    recommendation=(
                        "This is strong evidence to keep optimizing the selected engine. "
                        "Next target: profile the full workflow and move the hottest "
                        "remaining inner-loop pieces into engine methods."
                    ),
                )
            )
        elif workflow_speedup >= 0.9:
            findings.append(
                TriageFinding(
                    category="workflow",
                    severity="medium",
                    title="Workflow speedup is marginal",
                    evidence=f"Best non-reference workflow speedup: {workflow_speedup:.3g}x.",
                    recommendation=(
                        "Projection acceleration is not yet dominating end-to-end time. "
                        "Next target should be profiling residual assembly, ODE solving, "
                        "and optimizer callback overhead."
                    ),
                )
            )
        else:
            findings.append(
                TriageFinding(
                    category="workflow",
                    severity="high",
                    title="Optional engine is slower in full workflow",
                    evidence=f"Best non-reference workflow speedup: {workflow_speedup:.3g}x.",
                    recommendation=(
                        "Do not default to this engine for workflows yet. Keep it optional. "
                        "Focus on reducing overhead or move larger batches into the engine "
                        "before revisiting workflow acceleration."
                    ),
                )
            )

    if (
        kernel_speedup is not None
        and workflow_speedup is not None
        and kernel_speedup >= 1.5
        and workflow_speedup < 1.1
    ):
        findings.append(
            TriageFinding(
                category="bottleneck_inference",
                severity="high",
                title="Kernel speedup does not translate to workflow speedup",
                evidence=(
                    f"Kernel speedup is {kernel_speedup:.3g}x, but workflow "
                    f"speedup is only {workflow_speedup:.3g}x."
                ),
                recommendation=(
                    "The bottleneck is likely outside the scalar projection kernel. "
                    "Prioritize timing residual assembly, ODE simulation calls, "
                    "DataFrame construction, and scipy least_squares callback overhead."
                ),
            )
        )

    if workflow_speedup is None:
        findings.append(
            TriageFinding(
                category="benchmark_coverage",
                severity="medium",
                title="No full workflow benchmark found",
                evidence=(
                    "Could not find benchmark rows containing "
                    "'single_species_variable_projection_fit'."
                ),
                recommendation=(
                    "Run scripts/benchmark_variable_projection_engines.py and then "
                    "scripts/summarize_engine_benchmarks.py before making acceleration decisions."
                ),
            )
        )

    return findings


def findings_to_dataframe(findings: list[TriageFinding]) -> pd.DataFrame:
    return pd.DataFrame([asdict(finding) for finding in findings])


def make_markdown_report(
    findings: list[TriageFinding],
    *,
    summary_path: Path,
) -> str:
    lines = [
        "# Engine Optimization Triage",
        "",
        f"Input summary: `{summary_path}`",
        "",
        "## Findings",
        "",
    ]

    for index, finding in enumerate(findings, start=1):
        lines.extend(
            [
                f"### {index}. {finding.title}",
                "",
                f"- Category: `{finding.category}`",
                f"- Severity: `{finding.severity}`",
                f"- Evidence: {finding.evidence}",
                f"- Recommendation: {finding.recommendation}",
                "",
            ]
        )

    lines.extend(
        [
            "## Suggested next optimization order",
            "",
            "1. Confirm full workflow benchmark coverage.",
            "2. If kernel speedup does not translate to workflow speedup, profile callback overhead and residual assembly.",
            "3. If batch projection speedup is strong, wire larger batched operations into the workflow.",
            "4. Only move toward ODE/JAX/GPU solver work after projection and residual assembly are no longer obvious bottlenecks.",
            "",
        ]
    )

    return "\n".join(lines)


def write_triage_outputs(
    findings: list[TriageFinding],
    *,
    output_dir: Path,
    summary_path: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    dataframe = findings_to_dataframe(findings)

    csv_path = output_dir / "optimization_triage.csv"
    json_path = output_dir / "optimization_triage.json"
    markdown_path = output_dir / "optimization_triage.md"

    dataframe.to_csv(csv_path, index=False)

    with json_path.open("w") as handle:
        json.dump([asdict(finding) for finding in findings], handle, indent=2)

    markdown_path.write_text(
        make_markdown_report(
            findings,
            summary_path=summary_path,
        )
    )

    return {
        "csv": csv_path,
        "json": json_path,
        "markdown": markdown_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Triage engine benchmark results and suggest next optimization targets."
    )

    parser.add_argument(
        "--summary-csv",
        default="benchmarks/summary/engine_benchmark_summary.csv",
    )

    parser.add_argument(
        "--output-dir",
        default="benchmarks/triage",
    )

    args = parser.parse_args()

    summary_path = Path(args.summary_csv)
    dataframe = _load_summary(summary_path)

    findings = triage_benchmarks(dataframe)

    written_files = write_triage_outputs(
        findings,
        output_dir=Path(args.output_dir),
        summary_path=summary_path,
    )

    print("\nEngine optimization triage")
    print("==========================")
    print(f"Findings: {len(findings)}")

    for finding in findings:
        print(f"[{finding.severity}] {finding.title}")

    print("\nWritten files:")
    for name, path in written_files.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
