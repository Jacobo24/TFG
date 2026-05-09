import networkx as nx


def create_small_world_network(num_nodes: int, k: int, p: float, seed: int | None = None) -> nx.Graph:
    """
    Crea una red small-world de Watts-Strogatz.

    num_nodes:
        Número de individuos de la comunidad.

    k:
        Número de vecinos iniciales de cada nodo.
        Debe ser par y menor que num_nodes.

    p:
        Probabilidad de reconexión.

    seed:
        Semilla aleatoria.
    """
    if k >= num_nodes:
        raise ValueError("k debe ser menor que el número de nodos.")

    if k % 2 != 0:
        raise ValueError("k debe ser par en el modelo Watts-Strogatz.")

    return nx.watts_strogatz_graph(
        n=num_nodes,
        k=k,
        p=p,
        seed=seed
    )