from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_profile_likelihoods(
    profile_table: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    written_files: dict[str, Path] = {}

    for parameter, table in profile_table.groupby("parameter"):
        table = table.sort_values("fixed_value")

        fig, ax = plt.subplots()

        ax.plot(
            table["fixed_value"],
            table["delta_rss"],
            marker="o",
        )

        ax.set_xlabel(parameter)
        ax.set_ylabel("Delta RSS")
        ax.set_title(f"Profile likelihood: {parameter}")

        path = output_path / f"profile_likelihood_{parameter}.png"
        fig.savefig(path, bbox_inches="tight", dpi=300)
        plt.close(fig)

        written_files[f"profile_likelihood_{parameter}"] = path

    return written_files
