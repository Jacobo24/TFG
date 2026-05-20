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

def plot_b_sensitivity_final_cooperation(summary_df, save_path: str | None = None):
    """
    Grafica la cooperación final media en función del parámetro b.
    """
    plt.figure(figsize=(8, 5))

    plt.plot(
        summary_df["b"],
        summary_df["final_mean_cooperation"],
        marker="o",
        linewidth=2
    )

    plt.xlabel("Parámetro b")
    plt.ylabel("Cooperación final media")
    plt.title("Sensibilidad de la cooperación final respecto a b")
    plt.grid(True)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

def plot_social_adaptation_sensitivity(summary_df, save_path: str | None = None):
    """
    Grafica la cooperación final media según la tasa de adaptación social.
    """
    plt.figure(figsize=(8, 5))

    plt.plot(
        summary_df["social_adaptation_rate"],
        summary_df["final_mean_cooperation"],
        marker="o",
        linewidth=2
    )

    plt.xlabel("Tasa de adaptación social")
    plt.ylabel("Cooperación final media")
    plt.title("Sensibilidad a la inercia social")
    plt.grid(True)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

def plot_eta_b_heatmap(summary_df, save_path: str | None = None):
    """
    Representa la cooperación final media en función de b y eta_s.
    """
    pivot = summary_df.pivot(
        index="social_adaptation_rate",
        columns="b",
        values="mean_final_cooperation"
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    image = ax.imshow(pivot.values, aspect="auto", origin="lower")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{b:.2f}" for b in pivot.columns])

    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{eta:.3f}" for eta in pivot.index])

    ax.set_xlabel("Parámetro b")
    ax.set_ylabel("Tasa de adaptación social")
    ax.set_title("Cooperación final media según b y adaptación social")

    fig.colorbar(image, ax=ax, label="Cooperación final media")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_eta_b_lines(summary_df, save_path: str | None = None):
    """
    Representa la cooperación final media frente a eta_s,
    separando una curva por cada valor de b.
    """
    plt.figure(figsize=(8, 5))

    for b, df_group in summary_df.groupby("b"):
        df_group = df_group.sort_values("social_adaptation_rate")

        plt.plot(
            df_group["social_adaptation_rate"],
            df_group["mean_final_cooperation"],
            marker="o",
            linewidth=2,
            label=f"b={b:.2f}"
        )

    plt.xlabel("Tasa de adaptación social")
    plt.ylabel("Cooperación final media")
    plt.title("Sensibilidad conjunta a b y adaptación social")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

def plot_eta_b_heatmaps_by_tin(summary_df, save_dir: str):
    """
    Genera un heatmap de cooperación final media para cada valor de Tin.

    Eje X: b
    Eje Y: social_adaptation_rate
    Color: mean_final_cooperation
    """
    from pathlib import Path

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for Tin, df_tin in summary_df.groupby("Tin"):
        pivot = df_tin.pivot(
            index="social_adaptation_rate",
            columns="b",
            values="mean_final_cooperation"
        )

        fig, ax = plt.subplots(figsize=(9, 6))

        image = ax.imshow(
            pivot.values,
            aspect="auto",
            origin="lower"
        )

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{b:.2f}" for b in pivot.columns])

        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{eta:.3f}" for eta in pivot.index])

        ax.set_xlabel("Parámetro b")
        ax.set_ylabel("Tasa de adaptación social")
        ax.set_title(f"Cooperación final media | Tin = {Tin}")

        fig.colorbar(image, ax=ax, label="Cooperación final media")

        plt.tight_layout()

        save_path = save_dir / f"heatmap_eta_b_Tin_{Tin}.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

        print(f"Heatmap guardado en: {save_path}")

def plot_eta_b_lines_by_tin(summary_df, save_dir: str):
    """
    Genera una gráfica de líneas para cada Tin.

    Eje X: social_adaptation_rate
    Eje Y: mean_final_cooperation
    Una línea por cada b.
    """
    from pathlib import Path

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for Tin, df_tin in summary_df.groupby("Tin"):
        plt.figure(figsize=(9, 6))

        for b, df_group in df_tin.groupby("b"):
            df_group = df_group.sort_values("social_adaptation_rate")

            plt.plot(
                df_group["social_adaptation_rate"],
                df_group["mean_final_cooperation"],
                marker="o",
                linewidth=2,
                label=f"b={b:.2f}"
            )

        plt.xlabel("Tasa de adaptación social")
        plt.ylabel("Cooperación final media")
        plt.title(f"Sensibilidad a b y adaptación social | Tin = {Tin}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        save_path = save_dir / f"lines_eta_b_Tin_{Tin}.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

        print(f"Gráfica de líneas guardada en: {save_path}")