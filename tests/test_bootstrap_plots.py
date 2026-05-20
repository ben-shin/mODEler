import pandas as pd

from odefit.plotting.bootstrap_plots import (
    plot_bootstrap_parameter_histograms,
    plot_bootstrap_parameter_pairs,
)


def test_bootstrap_parameter_plots_are_written(tmp_path):
    parameter_samples = pd.DataFrame(
        {
            "bootstrap_index": [0, 1, 2, 3],
            "k1f": [0.1, 0.11, 0.09, 0.105],
            "k2f": [0.2, 0.21, 0.19, 0.205],
        }
    )

    histogram_files = plot_bootstrap_parameter_histograms(
        parameter_samples=parameter_samples,
        output_dir=tmp_path,
    )

    pair_files = plot_bootstrap_parameter_pairs(
        parameter_samples=parameter_samples,
        output_dir=tmp_path,
    )

    assert "histogram_k1f" in histogram_files
    assert "histogram_k2f" in histogram_files
    assert "pair_k1f_vs_k2f" in pair_files

    for path in list(histogram_files.values()) + list(pair_files.values()):
        assert path.exists()
        assert path.stat().st_size > 0
