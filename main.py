from src.config import ModelConfig
from src.model import MultilevelCooperationModel
from src.metrics import model_metrics_to_dataframe
from src.plots import plot_all_communities_cooperation


def main():
    config = ModelConfig()

    model = MultilevelCooperationModel(config)

    print("Comunidades antes de evolucionar:")
    print(model.community_summaries())

    model.run_internal_dynamics()

    print("\nComunidades después de evolucionar:")
    print(model.community_summaries())

    metrics_df = model_metrics_to_dataframe(model)

    print("\nPrimeras filas de métricas:")
    print(metrics_df.head())

    plot_all_communities_cooperation(metrics_df)


if __name__ == "__main__":
    main()