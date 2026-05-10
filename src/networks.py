import numpy as np
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


def create_intercommunity_network(
    sizes: list[int],
    base_prob: float = 0.25,
    seed: int | None = None
) -> nx.Graph:
    """
    Crea una red entre comunidades.

    Cada comunidad es un nodo. La probabilidad de conexión entre dos
    comunidades aumenta moderadamente con el tamaño de ambas.

    sizes:
        Lista con los tamaños de las comunidades.

    base_prob:
        Probabilidad base de conexión.

    seed:
        Semilla aleatoria.
    """
    rng = np.random.default_rng(seed)
    num_communities = len(sizes)

    graph = nx.Graph()
    graph.add_nodes_from(range(num_communities))

    sizes_array = np.array(sizes, dtype=float)

    min_size = sizes_array.min()
    max_size = sizes_array.max()

    if max_size == min_size:
        normalized_sizes = np.ones(num_communities)
    else:
        normalized_sizes = (sizes_array - min_size) / (max_size - min_size)

    for i in range(num_communities):
        for j in range(i + 1, num_communities):
            size_factor = 0.5 * (normalized_sizes[i] + normalized_sizes[j])

            connection_prob = base_prob * (1.0 + size_factor)
            connection_prob = min(connection_prob, 1.0)

            if rng.random() < connection_prob:
                graph.add_edge(i, j)

    return graph