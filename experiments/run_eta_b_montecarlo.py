from pathlib import Path

import pandas as pd

from src.config import ModelConfig
from src.model import MultilevelCooperationModel
from src.plots import plot_eta_b_heatmap, plot_eta_b_lines


def run_eta_b_montecarlo(
    eta_values: list[float],
    b_values: list[float],
    num_runs: int = 30,
    base_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ejecuta un experimento Monte Carlo variando:
    - b: tentación a desertar en el dilema del prisionero interno.
    - eta_s: tasa de adaptación social.

    Para ahorrar tiempo y espacio, solo guarda resultados finales.
    """
    raw_results = []

    for b in b_values:
        for eta in eta_values:
            print(f"Ejecutando b={b}, eta={eta}")

            for run_id in range(num_runs):
                seed = base_seed + run_id

                config = ModelConfig(
                    b=b,
                    social_adaptation_rate=eta,
                    seed=seed,
                )

                model = MultilevelCooperationModel(config)
                model.run_internal_dynamics()

                summary = model.community_summaries()

                raw_results.append({
                    "run_id": run_id,
                    "seed": seed,
                    "b": b,
                    "social_adaptation_rate": eta,
                    "final_mean_cooperation": summary["cooperation_rate"].mean(),
                    "final_std_cooperation_between_communities": summary["cooperation_rate"].std(),
                    "final_mean_payoff": summary["mean_payoff"].mean(),
                    "final_std_payoff_between_communities": summary["mean_payoff"].std(),
                })

    raw_df = pd.DataFrame(raw_results)

    summary_df = (
        raw_df
        .groupby(["b", "social_adaptation_rate"])
        .agg(
            mean_final_cooperation=("final_mean_cooperation", "mean"),
            std_final_cooperation=("final_mean_cooperation", "std"),
            mean_final_payoff=("final_mean_payoff", "mean"),
            std_final_payoff=("final_mean_payoff", "std"),
        )
        .reset_index()
    )

    return raw_df, summary_df


def main():
    data_dir = Path("results/data")
    figures_dir = Path("results/figures")

    data_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    eta_values = [0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25]
    b_values = [1.05, 1.10, 1.20, 1.30, 1.50]

    num_runs = 30

    raw_df, summary_df = run_eta_b_montecarlo(
        eta_values=eta_values,
        b_values=b_values,
        num_runs=num_runs,
        base_seed=42,
    )

    raw_path = data_dir / "eta_b_montecarlo_raw.csv"
    summary_path = data_dir / "eta_b_montecarlo_summary.csv"

    raw_df.to_csv(raw_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"\nResultados completos guardados en: {raw_path}")
    print(f"Resumen agregado guardado en: {summary_path}")

    print("\nResumen:")
    print(summary_df)

    heatmap_path = figures_dir / "eta_b_montecarlo_heatmap.png"
    lines_path = figures_dir / "eta_b_montecarlo_lines.png"

    plot_eta_b_heatmap(
        summary_df,
        save_path=str(heatmap_path),
    )

    plot_eta_b_lines(
        summary_df,
        save_path=str(lines_path),
    )

    print(f"Heatmap guardado en: {heatmap_path}")
    print(f"Gráfica de líneas guardada en: {lines_path}")


if __name__ == "__main__":
    main()