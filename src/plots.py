import matplotlib.pyplot as plt


def plot_all_communities_cooperation(metrics_df, save_path: str | None = None):
    """
    Grafica la evolución de la cooperación de todas las comunidades.

    Si se proporciona save_path, guarda la figura en esa ruta.
    """
    plt.figure(figsize=(9, 5))

    for community_id, df_group in metrics_df.groupby("community_id"):
        plt.plot(
            df_group["round"],
            df_group["cooperation_rate"],
            marker="o",
            markersize=2,
            linewidth=1.5,
            label=f"Comunidad {community_id}"
        )

    plt.xlabel("Ronda")
    plt.ylabel("Proporción de cooperadores")
    plt.title("Evolución de la cooperación por comunidad")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

import networkx as nx
import matplotlib.pyplot as plt


def plot_intercommunity_network(graph, external_df=None, save_path: str | None = None):
    """
    Grafica la red intercomunitaria.

    Si se proporciona external_df, etiqueta cada nodo con su comunidad
    y su probabilidad externa de cooperación.
    """
    plt.figure(figsize=(7, 5))

    pos = nx.spring_layout(graph, seed=42)

    if external_df is not None:
        labels = {
            row["community_id"]: f"C{int(row['community_id'])}\nP={row['external_cooperation_probability']:.2f}"
            for _, row in external_df.iterrows()
        }
    else:
        labels = {node: f"C{node}" for node in graph.nodes()}

    nx.draw_networkx_nodes(graph, pos, node_size=1200)
    nx.draw_networkx_edges(graph, pos, width=1.5)
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=9)

    plt.title("Red intercomunitaria")
    plt.axis("off")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

def plot_montecarlo_cooperation(summary_df, save_path: str | None = None):
    """
    Grafica la cooperación media por ronda en las simulaciones Monte Carlo,
    junto con una banda de desviación típica.
    """
    rounds = summary_df["round"]
    mean_coop = summary_df["mean_cooperation"]
    std_coop = summary_df["std_cooperation"]

    lower = mean_coop - std_coop
    upper = mean_coop + std_coop

    plt.figure(figsize=(9, 5))
    plt.plot(rounds, mean_coop, linewidth=2, label="Cooperación media")
    plt.fill_between(rounds, lower, upper, alpha=0.3, label="±1 desviación típica")

    plt.xlabel("Ronda")
    plt.ylabel("Proporción de cooperadores")
    plt.title("Evolución media de la cooperación (Monte Carlo)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

def plot_montecarlo_payoff(summary_df, save_path: str | None = None):
    """
    Grafica el payoff medio por ronda en las simulaciones Monte Carlo,
    junto con una banda de desviación típica.
    """
    rounds = summary_df["round"]
    mean_payoff = summary_df["mean_payoff"]
    std_payoff = summary_df["std_payoff"]

    lower = mean_payoff - std_payoff
    upper = mean_payoff + std_payoff

    plt.figure(figsize=(9, 5))
    plt.plot(rounds, mean_payoff, linewidth=2, label="Payoff medio")
    plt.fill_between(rounds, lower, upper, alpha=0.3, label="±1 desviación típica")

    plt.xlabel("Ronda")
    plt.ylabel("Payoff medio")
    plt.title("Evolución media del payoff (Monte Carlo)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()