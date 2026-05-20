from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd


@dataclass
class IdentifiabilityWarning:
    warning_type: str
    severity: str
    target: str
    message: str


@dataclass
class IdentifiabilityReport:
    warnings: list[IdentifiabilityWarning]

    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "warning_type": warning.warning_type,
                    "severity": warning.severity,
                    "target": warning.target,
                    "message": warning.message,
                }
                for warning in self.warnings
            ]
        )


def diagnose_bootstrap_parameter_summary(
    summary_table: pd.DataFrame,
    *,
    max_relative_ci_width: float = 2.0,
    max_cv: float = 1.0,
) -> list[IdentifiabilityWarning]:
    warnings: list[IdentifiabilityWarning] = []

    for _, row in summary_table.iterrows():
        parameter = str(row["parameter"])
        mean = float(row["mean"])
        std = float(row["std"])
        ci_lower = float(row["ci_lower"])
        ci_upper = float(row["ci_upper"])

        if np.isfinite(mean) and mean != 0:
            cv = abs(std / mean)

            if cv > max_cv:
                warnings.append(
                    IdentifiabilityWarning(
                        warning_type="high_bootstrap_cv",
                        severity="warning",
                        target=parameter,
                        message=(
                            f"Bootstrap coefficient of variation is high "
                            f"for {parameter}: CV={cv:.3g}."
                        ),
                    )
                )

            relative_ci_width = abs((ci_upper - ci_lower) / mean)

            if relative_ci_width > max_relative_ci_width:
                warnings.append(
                    IdentifiabilityWarning(
                        warning_type="wide_bootstrap_ci",
                        severity="warning",
                        target=parameter,
                        message=(
                            f"Bootstrap confidence interval is wide for "
                            f"{parameter}: relative width="
                            f"{relative_ci_width:.3g}."
                        ),
                    )
                )

        if not np.isfinite(ci_lower) or not np.isfinite(ci_upper):
            warnings.append(
                IdentifiabilityWarning(
                    warning_type="nonfinite_bootstrap_ci",
                    severity="error",
                    target=parameter,
                    message=(
                        f"Bootstrap confidence interval contains non-finite "
                        f"values for {parameter}."
                    ),
                )
            )

    return warnings


def diagnose_bootstrap_failures(
    *,
    n_requested: int,
    n_failed: int,
    max_failure_fraction: float = 0.1,
) -> list[IdentifiabilityWarning]:
    if n_requested <= 0:
        return []

    failure_fraction = n_failed / n_requested

    if failure_fraction <= max_failure_fraction:
        return []

    return [
        IdentifiabilityWarning(
            warning_type="high_bootstrap_failure_fraction",
            severity="warning",
            target="bootstrap",
            message=(
                f"Bootstrap failure fraction is high: "
                f"{n_failed}/{n_requested} "
                f"({failure_fraction:.1%})."
            ),
        )
    ]


def diagnose_profile_likelihood_table(
    profile_table: pd.DataFrame,
    *,
    flat_delta_rss_threshold: float = 1e-3,
    boundary_tolerance_fraction: float = 0.05,
) -> list[IdentifiabilityWarning]:
    warnings: list[IdentifiabilityWarning] = []

    if profile_table.empty:
        return [
            IdentifiabilityWarning(
                warning_type="empty_profile_likelihood",
                severity="error",
                target="profile_likelihood",
                message="Profile likelihood table is empty.",
            )
        ]

    for parameter, table in profile_table.groupby("parameter"):
        table = table.sort_values("fixed_value").copy()

        finite_table = table[
            np.isfinite(table["fixed_value"])
            & np.isfinite(table["delta_rss"])
        ]

        if finite_table.empty:
            warnings.append(
                IdentifiabilityWarning(
                    warning_type="nonfinite_profile_likelihood",
                    severity="error",
                    target=str(parameter),
                    message=(
                        f"Profile likelihood for {parameter} contains no "
                        "finite profile points."
                    ),
                )
            )
            continue

        delta_range = (
            float(finite_table["delta_rss"].max())
            - float(finite_table["delta_rss"].min())
        )

        if delta_range < flat_delta_rss_threshold:
            warnings.append(
                IdentifiabilityWarning(
                    warning_type="flat_profile_likelihood",
                    severity="warning",
                    target=str(parameter),
                    message=(
                        f"Profile likelihood is very flat for {parameter}. "
                        f"Delta RSS range={delta_range:.3g}."
                    ),
                )
            )

        min_index = finite_table["delta_rss"].idxmin()
        min_position = finite_table.index.get_loc(min_index)
        n_points = len(finite_table)

        boundary_count = max(
            1,
            int(np.ceil(boundary_tolerance_fraction * n_points)),
        )

        if min_position < boundary_count or min_position >= n_points - boundary_count:
            warnings.append(
                IdentifiabilityWarning(
                    warning_type="profile_minimum_near_boundary",
                    severity="warning",
                    target=str(parameter),
                    message=(
                        f"Profile likelihood minimum for {parameter} is near "
                        "the edge of the profile grid. Consider widening the "
                        "profile range."
                    ),
                )
            )

        failed_points = int((~table["success"].astype(bool)).sum())

        if failed_points > 0:
            warnings.append(
                IdentifiabilityWarning(
                    warning_type="profile_fit_failures",
                    severity="warning",
                    target=str(parameter),
                    message=(
                        f"Profile likelihood for {parameter} has "
                        f"{failed_points} failed grid points."
                    ),
                )
            )

    return warnings


def diagnose_parameters_near_bounds(
    fitted_parameters: dict[str, float],
    parameter_bounds: dict[str, tuple[float, float]],
    *,
    relative_tolerance: float = 0.01,
) -> list[IdentifiabilityWarning]:
    warnings: list[IdentifiabilityWarning] = []

    for name, value in fitted_parameters.items():
        if name not in parameter_bounds:
            continue

        lower, upper = parameter_bounds[name]
        lower = float(lower)
        upper = float(upper)
        value = float(value)

        width = upper - lower

        if width <= 0:
            continue

        lower_distance = abs(value - lower) / width
        upper_distance = abs(upper - value) / width

        if lower_distance <= relative_tolerance:
            warnings.append(
                IdentifiabilityWarning(
                    warning_type="parameter_near_lower_bound",
                    severity="warning",
                    target=name,
                    message=(
                        f"Parameter {name} is close to its lower bound: "
                        f"value={value:g}, lower={lower:g}."
                    ),
                )
            )

        if upper_distance <= relative_tolerance:
            warnings.append(
                IdentifiabilityWarning(
                    warning_type="parameter_near_upper_bound",
                    severity="warning",
                    target=name,
                    message=(
                        f"Parameter {name} is close to its upper bound: "
                        f"value={value:g}, upper={upper:g}."
                    ),
                )
            )

    return warnings


def build_identifiability_report(
    *,
    bootstrap_summary_table: pd.DataFrame | None = None,
    profile_likelihood_table: pd.DataFrame | None = None,
    fitted_parameters: dict[str, float] | None = None,
    parameter_bounds: dict[str, tuple[float, float]] | None = None,
    n_bootstrap_requested: int | None = None,
    n_bootstrap_failed: int | None = None,
) -> IdentifiabilityReport:
    warnings: list[IdentifiabilityWarning] = []

    if bootstrap_summary_table is not None:
        warnings.extend(
            diagnose_bootstrap_parameter_summary(
                bootstrap_summary_table,
            )
        )

    if (
        n_bootstrap_requested is not None
        and n_bootstrap_failed is not None
    ):
        warnings.extend(
            diagnose_bootstrap_failures(
                n_requested=n_bootstrap_requested,
                n_failed=n_bootstrap_failed,
            )
        )

    if profile_likelihood_table is not None:
        warnings.extend(
            diagnose_profile_likelihood_table(
                profile_likelihood_table,
            )
        )

    if fitted_parameters is not None and parameter_bounds is not None:
        warnings.extend(
            diagnose_parameters_near_bounds(
                fitted_parameters=fitted_parameters,
                parameter_bounds=parameter_bounds,
            )
        )

    return IdentifiabilityReport(warnings=warnings)


def export_identifiability_report(
    report: IdentifiabilityReport,
    output_dir: str | Path,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    written_files: dict[str, Path] = {}

    table = report.to_dataframe()

    csv_path = output_path / "identifiability_warnings.csv"
    table.to_csv(csv_path, index=False)
    written_files["identifiability_warnings"] = csv_path

    json_path = output_path / "identifiability_warnings.json"

    with json_path.open("w") as handle:
        json.dump(
            {
                "has_warnings": report.has_warnings,
                "n_warnings": len(report.warnings),
                "warnings": table.to_dict(orient="records"),
            },
            handle,
            indent=2,
        )

    written_files["identifiability_warnings_json"] = json_path

    return written_files
