from pathlib import Path

import pandas as pd

from src.config import ModelConfig
from src.model import MultilevelCooperationModel
from src.metrics import model_metrics_to_dataframe
from src.plots import plot_montecarlo_cooperation, plot_montecarlo_payoff
# python -m experiments.run_montecarlo_internal

def run_montecarlo_internal(num_runs: int = 50, base_seed: int = 42) -> pd.DataFrame:
    """
    Ejecuta varias simulaciones Monte Carlo de la dinámica intracomunitaria.

    Cada simulación usa una semilla distinta y devuelve la evolución temporal
    de todas las comunidades.
    """
    all_results = []

    for run_id in range(num_runs):
        config = ModelConfig(seed=base_seed + run_id)

        model = MultilevelCooperationModel(config)
        model.run_internal_dynamics()

        metrics_df = model_metrics_to_dataframe(model)
        metrics_df["run_id"] = run_id

        all_results.append(metrics_df)

    return pd.concat(all_results, ignore_index=True)


def main():
    data_dir = Path("results/data")
    data_dir.mkdir(parents=True, exist_ok=True)

    num_runs = 50
    results_df = run_montecarlo_internal(num_runs=num_runs)

    output_path = data_dir / "montecarlo_internal_dynamics.csv"
    results_df.to_csv(output_path, index=False)

    summary_df = (
        results_df
        .groupby("round")
        .agg(
            mean_cooperation=("cooperation_rate", "mean"),
            std_cooperation=("cooperation_rate", "std"),
            mean_payoff=("mean_payoff", "mean"),
            std_payoff=("mean_payoff", "std")
        )
        .reset_index()
    )

    summary_path = data_dir / "montecarlo_internal_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    figures_dir = Path("results/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    figure_path = figures_dir / "montecarlo_internal_cooperation.png"

    plot_montecarlo_cooperation(
        summary_df,
        save_path=str(figure_path)
    )

    payoff_figure_path = figures_dir / "montecarlo_internal_payoff.png"

    plot_montecarlo_payoff(
        summary_df,
        save_path=str(payoff_figure_path)
    )

    print(f"Gráfica Monte Carlo de payoff guardada en: {payoff_figure_path}")

    print(f"Gráfica Monte Carlo guardada en: {figure_path}")

    print(f"Resumen Monte Carlo guardado en: {summary_path}")
    print(summary_df.head())

    print(f"Resultados Monte Carlo guardados en: {output_path}")
    print(results_df.head())


if __name__ == "__main__":
    main()