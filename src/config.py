from dataclasses import dataclass


@dataclass
class ModelConfig:
    # Número de comunidades
    num_communities: int = 5

    # Tamaños de las comunidades
    min_community_size: int = 50
    max_community_size: int = 100

    # Red small-world interna
    small_world_k: int = 6
    small_world_p: float = 0.1

    # Dilema del prisionero débil
    b: float = 1.5

    # Condición inicial
    xmin: float = 0.4
    xmax: float = 0.5

    # Memoria temporal
    memory_length: int = 3

    # Rondas internas
    Tin: int = 100

    # Pesos de la propensión social
    alpha_I: float = 0.35
    alpha_D: float = 0.30
    alpha_R: float = 0.20
    alpha_E: float = 0.15

    # Errores
    epsilon_execution: float = 0.01

    # Dinámica Fermi
    lambda_fermi: float = 1.0
    delta_D: float = 0.5
    delta_pi: float = 0.5

    # Nivel intercomunitario
    beta_x: float = 0.7
    beta_pi: float = 0.3

    # Semilla aleatoria
    seed: int = 42