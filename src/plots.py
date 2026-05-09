import matplotlib.pyplot as plt


def plot_community_cooperation(metrics_df, community_id: int | None = None):
    """
    Grafica la evolución de la cooperación de una comunidad.
    """
    df = metrics_df.copy()

    if community_id is not None:
        df = df[df["community_id"] == community_id]

    plt.figure(figsize=(8, 5))
    plt.plot(df["round"], df["cooperation_rate"], marker="o", markersize=3)
    plt.xlabel("Ronda")
    plt.ylabel("Proporción de cooperadores")
    plt.title("Evolución de la cooperación")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_all_communities_cooperation(metrics_df):
    """
    Grafica la evolución de la cooperación de todas las comunidades.
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
    plt.show()