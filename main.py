import pandas as pd
from pathlib import Path

from src.config import ModelConfig
from src.model import MultilevelCooperationModel
from src.metrics import model_metrics_to_dataframe
from src.plots import plot_all_communities_cooperation, plot_intercommunity_network

def main():
    config = ModelConfig()

    data_dir = Path("results/data")
    figures_dir = Path("results/figures")

    data_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    model = MultilevelCooperationModel(config)

    print("Comunidades antes de evolucionar:")
    print(model.community_summaries())

    model.run_internal_dynamics()

    print("\nComunidades después de evolucionar:")
    summary_df = model.community_summaries()
    print(summary_df)

    summary_path = data_dir / "baseline_community_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    metrics_df = model_metrics_to_dataframe(model)

    csv_path = data_dir / "baseline_internal_dynamics.csv"
    metrics_df.to_csv(csv_path, index=False)

    figure_path = figures_dir / "baseline_internal_cooperation.png"

    plot_all_communities_cooperation(
        metrics_df,
        save_path=str(figure_path)
    )

    print(f"\nResultados internos guardados en: {csv_path}")
    print(f"Resumen de comunidades guardado en: {summary_path}")
    print(f"Figura interna guardada en: {figure_path}")

    # Nivel intercomunitario
    model.build_intercommunity_network()

    external_df, edge_df = model.run_intercommunity_interactions()

    intercommunity_figure_path = figures_dir / "baseline_intercommunity_network.png"

    plot_intercommunity_network(
        model.intercommunity_graph,
        external_df=external_df,
        save_path=str(intercommunity_figure_path)
    )

    print(f"Figura de red intercomunitaria guardada en: {intercommunity_figure_path}")

    print("\nResultados por comunidad en el nivel intercomunitario:")
    print(external_df)

    print("\nInteracciones entre comunidades:")
    print(edge_df)

    expected_global = model.expected_global_cooperation()
    observed_global = model.observed_global_cooperation(external_df)

    global_summary_df = pd.DataFrame([{
        "expected_global_cooperation": expected_global,
        "observed_global_cooperation": observed_global,
        "num_communities": len(model.communities),
        "num_intercommunity_edges": model.intercommunity_graph.number_of_edges()
    }])

    global_summary_path = data_dir / "baseline_global_summary.csv"
    global_summary_df.to_csv(global_summary_path, index=False)

    print(f"Resumen global guardado en: {global_summary_path}")

    print("\nCooperación global esperada:")
    print(expected_global)

    print("\nCooperación global observada:")
    print(observed_global)

    external_path = data_dir / "baseline_external_communities.csv"
    edges_path = data_dir / "baseline_external_edges.csv"

    external_df.to_csv(external_path, index=False)
    edge_df.to_csv(edges_path, index=False)

    print(f"\nResultados externos por comunidad guardados en: {external_path}")
    print(f"Interacciones externas guardadas en: {edges_path}")


if __name__ == "__main__":
    main()