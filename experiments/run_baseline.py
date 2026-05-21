from pathlib import Path

import pandas as pd

from src.config import ModelConfig
from src.model import MultilevelCooperationModel
from src.metrics import model_metrics_to_dataframe
from src.plots import (
    plot_all_communities_cooperation,
    plot_intercommunity_network,
)


def run_baseline():
    """
    Ejecuta una simulación baseline del modelo completo.

    Esta versión se usa para pruebas pequeñas y controladas:
    - dinámica interna de comunidades;
    - red intercomunitaria;
    - cooperación externa;
    - guardado de resultados y figuras.
    """

    config = ModelConfig(
        # Parámetros principales
        b=2.0,
        Tin=1000,

        # Estado inicial
        xmin=0.4,
        xmax=0.5,

        # Adaptación social
        social_adaptation_rate=0.20,

        # Revisión evolutiva
        delta_D=0.05,
        delta_pi=0.02,
        lambda_fermi=1.0,

        # Semilla
        seed=42,
    )

    data_dir = Path("results/data/baseline")
    figures_dir = Path("results/figures/baseline")

    data_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    model = MultilevelCooperationModel(config)

    print("Comunidades antes de evolucionar:")
    initial_summary_df = model.community_summaries()
    print(initial_summary_df)

    model.run_internal_dynamics()

    print("\nComunidades después de evolucionar:")
    final_summary_df = model.community_summaries()
    print(final_summary_df)

    # Guardar resumen final de comunidades
    community_summary_path = data_dir / "baseline_community_summary.csv"
    final_summary_df.to_csv(community_summary_path, index=False)

    # Guardar dinámica interna
    metrics_df = model_metrics_to_dataframe(model)

    internal_dynamics_path = data_dir / "baseline_internal_dynamics.csv"
    metrics_df.to_csv(internal_dynamics_path, index=False)

    # Figura de cooperación interna
    internal_figure_path = figures_dir / "baseline_internal_cooperation.png"

    plot_all_communities_cooperation(
        metrics_df,
        save_path=str(internal_figure_path),
    )

    # Nivel intercomunitario
    model.build_intercommunity_network()

    external_df, edge_df = model.run_intercommunity_interactions()

    print("\nResultados por comunidad en el nivel intercomunitario:")
    print(external_df)

    print("\nInteracciones entre comunidades:")
    print(edge_df)

    # Figura de red intercomunitaria
    intercommunity_figure_path = figures_dir / "baseline_intercommunity_network.png"

    plot_intercommunity_network(
        model.intercommunity_graph,
        external_df=external_df,
        save_path=str(intercommunity_figure_path),
    )

    # Cooperación global
    expected_global = model.expected_global_cooperation()
    observed_global = model.observed_global_cooperation(external_df)

    global_summary_df = pd.DataFrame([{
        "b": config.b,
        "Tin": config.Tin,
        "xmin": config.xmin,
        "xmax": config.xmax,
        "social_adaptation_rate": config.social_adaptation_rate,
        "delta_D": config.delta_D,
        "delta_pi": config.delta_pi,
        "lambda_fermi": config.lambda_fermi,
        "expected_global_cooperation": expected_global,
        "observed_global_cooperation": observed_global,
        "num_communities": len(model.communities),
        "num_intercommunity_edges": model.intercommunity_graph.number_of_edges(),
    }])

    global_summary_path = data_dir / "baseline_global_summary.csv"
    global_summary_df.to_csv(global_summary_path, index=False)

    # Guardar resultados externos
    external_path = data_dir / "baseline_external_communities.csv"
    edges_path = data_dir / "baseline_external_edges.csv"

    external_df.to_csv(external_path, index=False)
    edge_df.to_csv(edges_path, index=False)

    print(f"\nDinámica interna guardada en: {internal_dynamics_path}")
    print(f"Resumen de comunidades guardado en: {community_summary_path}")
    print(f"Figura interna guardada en: {internal_figure_path}")
    print(f"Figura intercomunitaria guardada en: {intercommunity_figure_path}")
    print(f"Resumen global guardado en: {global_summary_path}")
    print(f"Resultados externos por comunidad guardados en: {external_path}")
    print(f"Interacciones externas guardadas en: {edges_path}")

    print("\nCooperación global esperada:")
    print(expected_global)

    print("\nCooperación global observada:")
    print(observed_global)


def main():
    run_baseline()


if __name__ == "__main__":
    main()