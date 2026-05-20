from pathlib import Path

import pandas as pd

from src.config import ModelConfig
from src.model import MultilevelCooperationModel
from src.metrics import model_metrics_to_dataframe
from src.plots import plot_social_adaptation_sensitivity


def run_social_adaptation_sensitivity(
    eta_values: list[float],
    b: float = 1.2,
    base_seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ejecuta el modelo para distintos valores de la tasa de adaptación social.

    eta = social_adaptation_rate

    Devuelve:
    - all_dynamics_df: evolución temporal para cada valor de eta.
    - summary_df: resumen final agregado para cada valor de eta.
    """
    all_dynamics = []
    summaries = []

    for eta in eta_values:
        config = ModelConfig(
            b=b,
            social_adaptation_rate=eta,
            seed=base_seed
        )

        model = MultilevelCooperationModel(config)
        model.run_internal_dynamics()

        dynamics_df = model_metrics_to_dataframe(model)
        dynamics_df["social_adaptation_rate"] = eta
        all_dynamics.append(dynamics_df)

        community_summary = model.community_summaries()

        summaries.append({
            "social_adaptation_rate": eta,
            "final_mean_cooperation": community_summary["cooperation_rate"].mean(),
            "final_std_cooperation": community_summary["cooperation_rate"].std(),
            "final_mean_payoff": community_summary["mean_payoff"].mean(),
            "final_std_payoff": community_summary["mean_payoff"].std()
        })

    all_dynamics_df = pd.concat(all_dynamics, ignore_index=True)
    summary_df = pd.DataFrame(summaries)

    return all_dynamics_df, summary_df


def main():
    data_dir = Path("results/data")
    figures_dir = Path("results/figures")

    data_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    eta_values = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.00]

    dynamics_df, summary_df = run_social_adaptation_sensitivity(
        eta_values=eta_values,
        b=1.2,
        base_seed=42
    )

    dynamics_path = data_dir / "social_adaptation_sensitivity_dynamics.csv"
    summary_path = data_dir / "social_adaptation_sensitivity_summary.csv"

    dynamics_df.to_csv(dynamics_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"Dinámica guardada en: {dynamics_path}")
    print(f"Resumen guardado en: {summary_path}")

    print("\nResumen final:")
    print(summary_df)

    figure_path = figures_dir / "social_adaptation_sensitivity.png"

    plot_social_adaptation_sensitivity(
        summary_df,
        save_path=str(figure_path)
    )

    print(f"Figura guardada en: {figure_path}")


if __name__ == "__main__":
    main()