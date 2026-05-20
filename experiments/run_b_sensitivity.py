from pathlib import Path

import pandas as pd

from src.config import ModelConfig
from src.model import MultilevelCooperationModel
from src.metrics import model_metrics_to_dataframe
from src.plots import plot_b_sensitivity_final_cooperation


def run_b_sensitivity(b_values: list[float], base_seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ejecuta el modelo para distintos valores del parámetro b.

    Devuelve:
    - all_dynamics_df: evolución temporal de todas las comunidades para cada b.
    - summary_df: resumen final agregado para cada b.
    """
    all_dynamics = []
    summaries = []

    for b in b_values:
        config = ModelConfig(
            b=b,
            seed=base_seed
        )

        model = MultilevelCooperationModel(config)

        model.run_internal_dynamics()

        dynamics_df = model_metrics_to_dataframe(model)
        dynamics_df["b"] = b
        all_dynamics.append(dynamics_df)

        community_summary = model.community_summaries()

        final_mean_cooperation = community_summary["cooperation_rate"].mean()
        final_std_cooperation = community_summary["cooperation_rate"].std()

        final_mean_payoff = community_summary["mean_payoff"].mean()
        final_std_payoff = community_summary["mean_payoff"].std()

        summaries.append({
            "b": b,
            "final_mean_cooperation": final_mean_cooperation,
            "final_std_cooperation": final_std_cooperation,
            "final_mean_payoff": final_mean_payoff,
            "final_std_payoff": final_std_payoff
        })

    all_dynamics_df = pd.concat(all_dynamics, ignore_index=True)
    summary_df = pd.DataFrame(summaries)

    return all_dynamics_df, summary_df


def main():
    data_dir = Path("results/data")
    figures_dir = Path("results/figures")

    data_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    b_values = [1.01, 1.05, 1.10, 1.20, 1.30, 1.40, 1.50, 1.70, 2.00]

    dynamics_df, summary_df = run_b_sensitivity(
        b_values=b_values,
        base_seed=42
    )

    dynamics_path = data_dir / "b_sensitivity_internal_dynamics.csv"
    summary_path = data_dir / "b_sensitivity_summary.csv"

    dynamics_df.to_csv(dynamics_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"Dinámica por b guardada en: {dynamics_path}")
    print(f"Resumen por b guardado en: {summary_path}")

    print("\nResumen final:")
    print(summary_df)

    figure_path = figures_dir / "b_sensitivity_final_cooperation.png"

    plot_b_sensitivity_final_cooperation(
        summary_df,
        save_path=str(figure_path)
    )

    print(f"Figura guardada en: {figure_path}")


if __name__ == "__main__":
    main()