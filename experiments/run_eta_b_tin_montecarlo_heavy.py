from pathlib import Path

import pandas as pd

from src.config import ModelConfig
from src.model import MultilevelCooperationModel


def run_eta_b_tin_montecarlo_heavy(
    eta_values: list[float],
    b_values: list[float],
    tin_values: list[int],
    num_runs: int = 100,
    base_seed: int = 42,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """
    Experimento Monte Carlo pesado variando:
    - b: tentación a desertar
    - eta_s: tasa de adaptación social
    - Tin: número de rondas internas

    Guarda solo resultados finales para reducir tamaño.
    Si output_path existe, reanuda evitando simulaciones ya calculadas.
    """

    existing_df = pd.DataFrame()

    if output_path is not None and output_path.exists():
        existing_df = pd.read_csv(output_path)
        print(f"Archivo existente encontrado. Reanudando desde: {output_path}")
        print(f"Filas ya calculadas: {len(existing_df)}")

    existing_keys = set()

    if not existing_df.empty:
        existing_keys = set(
            zip(
                existing_df["b"],
                existing_df["social_adaptation_rate"],
                existing_df["Tin"],
                existing_df["run_id"],
            )
        )

    new_results = []

    total_jobs = len(b_values) * len(eta_values) * len(tin_values) * num_runs
    completed = len(existing_keys)

    for Tin in tin_values:
        for b in b_values:
            for eta in eta_values:
                print(f"\nCalculando Tin={Tin}, b={b}, eta={eta}")

                for run_id in range(num_runs):
                    key = (b, eta, Tin, run_id)

                    if key in existing_keys:
                        continue

                    seed = base_seed + run_id

                    config = ModelConfig(
                        b=b,
                        Tin=Tin,
                        social_adaptation_rate=eta,
                        seed=seed,
                    )

                    model = MultilevelCooperationModel(config)
                    model.run_internal_dynamics()

                    community_summary = model.community_summaries()

                    final_mean_cooperation = community_summary["cooperation_rate"].mean()
                    final_std_cooperation = community_summary["cooperation_rate"].std()
                    final_mean_payoff = community_summary["mean_payoff"].mean()
                    final_std_payoff = community_summary["mean_payoff"].std()

                    new_results.append({
                        "run_id": run_id,
                        "seed": seed,
                        "Tin": Tin,
                        "b": b,
                        "social_adaptation_rate": eta,
                        "final_mean_cooperation": final_mean_cooperation,
                        "final_std_cooperation_between_communities": final_std_cooperation,
                        "final_mean_payoff": final_mean_payoff,
                        "final_std_payoff_between_communities": final_std_payoff,
                    })

                    completed += 1

                    if completed % 50 == 0:
                        print(f"Progreso aproximado: {completed}/{total_jobs}")

                    # Guardado parcial cada 100 simulaciones nuevas
                    if output_path is not None and len(new_results) % 100 == 0:
                        partial_df = pd.DataFrame(new_results)

                        if existing_df.empty:
                            save_df = partial_df
                        else:
                            save_df = pd.concat([existing_df, partial_df], ignore_index=True)

                        save_df.to_csv(output_path, index=False)

    new_df = pd.DataFrame(new_results)

    if existing_df.empty:
        final_df = new_df
    else:
        final_df = pd.concat([existing_df, new_df], ignore_index=True)

    if output_path is not None:
        final_df.to_csv(output_path, index=False)

    return final_df


def summarize_results(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Resume los resultados por combinación de Tin, b y eta_s.
    """
    summary_df = (
        raw_df
        .groupby(["Tin", "b", "social_adaptation_rate"])
        .agg(
            mean_final_cooperation=("final_mean_cooperation", "mean"),
            std_final_cooperation=("final_mean_cooperation", "std"),
            mean_final_payoff=("final_mean_payoff", "mean"),
            std_final_payoff=("final_mean_payoff", "std"),
        )
        .reset_index()
    )

    return summary_df


def main():
    data_dir = Path("results/data")
    figures_dir = Path("results/figures")

    data_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Valores centrados en la zona interesante que ya hemos detectado
    eta_values = [
        0.020,
        0.030,
        0.040,
        0.050,
        0.075,
        0.100,
        0.125,
        0.150,
        0.175,
        0.200,
        0.250,
    ]

    b_values = [
        1.01,
        1.05,
        1.10,
        1.15,
        1.20,
        1.30,
        1.40,
        1.50,
        1.70,
    ]

    tin_values = [
        50,
        100,
        150,
    ]

    num_runs = 80

    raw_path = data_dir / "eta_b_tin_montecarlo_heavy_raw.csv"
    summary_path = data_dir / "eta_b_tin_montecarlo_heavy_summary.csv"

    raw_df = run_eta_b_tin_montecarlo_heavy(
        eta_values=eta_values,
        b_values=b_values,
        tin_values=tin_values,
        num_runs=num_runs,
        base_seed=42,
        output_path=raw_path,
    )

    summary_df = summarize_results(raw_df)
    summary_df.to_csv(summary_path, index=False)

    print(f"\nResultados completos guardados en: {raw_path}")
    print(f"Resumen guardado en: {summary_path}")

    print("\nResumen:")
    print(summary_df)


if __name__ == "__main__":
    main()