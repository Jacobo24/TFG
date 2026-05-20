from pathlib import Path

import pandas as pd

from src.plots import (
    plot_eta_b_heatmaps_by_tin,
    plot_eta_b_lines_by_tin,
)


def main():
    data_path = Path("results/data/eta_b_tin_montecarlo_heavy_summary.csv")
    figures_dir = Path("results/figures/heavy_eta_b_tin")

    summary_df = pd.read_csv(data_path)

    plot_eta_b_heatmaps_by_tin(
        summary_df,
        save_dir=str(figures_dir)
    )

    plot_eta_b_lines_by_tin(
        summary_df,
        save_dir=str(figures_dir)
    )


if __name__ == "__main__":
    main()