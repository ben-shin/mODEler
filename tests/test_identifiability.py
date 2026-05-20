import pandas as pd

from odefit.fitting.identifiability import (
    build_identifiability_report,
    diagnose_bootstrap_parameter_summary,
    diagnose_parameters_near_bounds,
    diagnose_profile_likelihood_table,
    export_identifiability_report,
)


def test_diagnose_bootstrap_parameter_summary_flags_wide_ci():
    summary = pd.DataFrame(
        {
            "parameter": ["k1f"],
            "n": [20],
            "mean": [1.0],
            "std": [2.0],
            "median": [1.0],
            "ci_lower": [-5.0],
            "ci_upper": [7.0],
            "confidence_level": [0.95],
        }
    )

    warnings = diagnose_bootstrap_parameter_summary(summary)

    warning_types = {warning.warning_type for warning in warnings}

    assert "high_bootstrap_cv" in warning_types
    assert "wide_bootstrap_ci" in warning_types


def test_diagnose_profile_likelihood_flags_flat_profile():
    profile_table = pd.DataFrame(
        {
            "parameter": ["k1f"] * 5,
            "fixed_value": [0.1, 0.2, 0.3, 0.4, 0.5],
            "delta_rss": [0.0, 0.0001, 0.0002, 0.0001, 0.0],
            "success": [True, True, True, True, True],
        }
    )

    warnings = diagnose_profile_likelihood_table(
        profile_table,
        flat_delta_rss_threshold=1e-2,
    )

    warning_types = {warning.warning_type for warning in warnings}

    assert "flat_profile_likelihood" in warning_types


def test_diagnose_parameters_near_bounds():
    warnings = diagnose_parameters_near_bounds(
        fitted_parameters={"k1f": 0.001},
        parameter_bounds={"k1f": (0.0, 1.0)},
        relative_tolerance=0.01,
    )

    assert len(warnings) == 1
    assert warnings[0].warning_type == "parameter_near_lower_bound"


def test_build_and_export_identifiability_report(tmp_path):
    bootstrap_summary = pd.DataFrame(
        {
            "parameter": ["k1f"],
            "n": [20],
            "mean": [1.0],
            "std": [2.0],
            "median": [1.0],
            "ci_lower": [-5.0],
            "ci_upper": [7.0],
            "confidence_level": [0.95],
        }
    )

    report = build_identifiability_report(
        bootstrap_summary_table=bootstrap_summary,
        fitted_parameters={"k1f": 0.001},
        parameter_bounds={"k1f": (0.0, 1.0)},
    )

    assert report.has_warnings

    written_files = export_identifiability_report(
        report=report,
        output_dir=tmp_path,
    )

    assert written_files["identifiability_warnings"].exists()
    assert written_files["identifiability_warnings_json"].exists()
