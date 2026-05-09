import numpy as np
import networkx as nx

from src.config import ModelConfig
from src.networks import create_small_world_network


class Community:
    def __init__(self, community_id: int, size: int, config: ModelConfig, seed: int | None = None):
        self.community_id = community_id
        self.size = size
        self.config = config
        self.rng = np.random.default_rng(seed)

        self.graph = create_small_world_network(
            num_nodes=size,
            k=config.small_world_k,
            p=config.small_world_p,
            seed=seed
        )

        self.strategies = self._initialize_strategies()
        self.history = [self.strategies.copy()]
        self.payoffs = np.zeros(self.size)

        self.metrics_history = []
        self._save_metrics(round_number=0)

    def _initialize_strategies(self) -> np.ndarray:
        """
        Inicializa una proporción aleatoria de cooperadores entre xmin y xmax.
        """
        x0 = self.rng.uniform(self.config.xmin, self.config.xmax)
        num_cooperators = int(round(x0 * self.size))

        strategies = np.zeros(self.size, dtype=int)
        cooperators = self.rng.choice(self.size, size=num_cooperators, replace=False)
        strategies[cooperators] = 1

        return strategies
    

    def _payoff_pair(self, s_i: int, s_j: int) -> float:
        """
        Pago de i al interactuar con j.
        """
        if s_i == 1 and s_j == 1:
            return 1.0
        if s_i == 1 and s_j == 0:
            return 0.0
        if s_i == 0 and s_j == 1:
            return self.config.b
        return 0.0

    def compute_payoffs(self) -> np.ndarray:
        """
        Calcula el payoff acumulado de cada individuo como suma
        de sus interacciones con vecinos.
        """
        payoffs = np.zeros(self.size)

        for i in range(self.size):
            for j in self.graph.neighbors(i):
                payoffs[i] += self._payoff_pair(self.strategies[i], self.strategies[j])

        self.payoffs = payoffs
        return payoffs
    
    def _get_recent_history(self) -> np.ndarray:
        """
        Devuelve las últimas L estrategias disponibles.
        Si todavía no hay L rondas, usa las disponibles.
        """
        L = self.config.memory_length
        recent = self.history[-L:]
        return np.array(recent)
    
    def compute_social_components(self):
        """
        Calcula I, D, R y E para todos los individuos.
        """
        recent_history = self._get_recent_history()

        I = np.mean(recent_history, axis=0)

        D = np.zeros(self.size)
        R = np.zeros(self.size)
        E = np.zeros(self.size)

        reputations = np.mean(recent_history, axis=0)

        for i in range(self.size):
            neighbors = list(self.graph.neighbors(i))

            if len(neighbors) == 0:
                D[i] = 0.0
                R[i] = 0.0
                E[i] = 0.0
                continue

            # Entorno local: cooperación media reciente de los vecinos
            E[i] = np.mean(recent_history[:, neighbors])

            # Reputación media de los vecinos
            R[i] = np.mean(reputations[neighbors])

            # Reciprocidad directa: cooperación mutua reciente con vecinos
            mutual_cooperation = []

            for past_strategies in recent_history:
                for j in neighbors:
                    mutual_cooperation.append(past_strategies[i] * past_strategies[j])

            D[i] = np.mean(mutual_cooperation)

        return I, D, R, E
    
    def compute_social_propensity(self) -> np.ndarray:
        """
        Calcula la propensión social-relacional a cooperar.
        """
        I, D, R, E = self.compute_social_components()

        p_social = (
            self.config.alpha_I * I
            + self.config.alpha_D * D
            + self.config.alpha_R * R
            + self.config.alpha_E * E
        )

        return np.clip(p_social, 0.0, 1.0), I, D, R, E

    def normalize_payoffs(self) -> np.ndarray:
        """
        Normaliza los payoffs al intervalo [0, 1].

        Si todos los individuos tienen el mismo payoff, se devuelve un vector de unos
        para evitar que todos revisen su estrategia por un problema artificial
        de normalización.
        """
        min_payoff = np.min(self.payoffs)
        max_payoff = np.max(self.payoffs)

        if max_payoff == min_payoff:
            return np.ones(self.size)

        return (self.payoffs - min_payoff) / (max_payoff - min_payoff)

    def compute_revision_probability(self, D: np.ndarray) -> np.ndarray:
        """
        Calcula la probabilidad de que cada individuo revise su estrategia.
        """
        normalized_payoffs = self.normalize_payoffs()

        rho = (
            self.config.delta_D * (1.0 - D)
            + self.config.delta_pi * (1.0 - normalized_payoffs)
        )

        return np.clip(rho, 0.0, 1.0)
    
    def fermi_probability(self, i: int, j: int) -> float:
        """
        Probabilidad de que i adopte la estrategia de j según Fermi.
        """
        diff = self.payoffs[j] - self.payoffs[i]
        exponent = -self.config.lambda_fermi * diff

        return 1.0 / (1.0 + np.exp(exponent))
    
    def step(self):
        """
        Ejecuta una ronda completa de actualización síncrona.
        """
        self.compute_payoffs()

        p_social, I, D, R, E = self.compute_social_propensity()
        rho = self.compute_revision_probability(D)

        new_strategies = self.strategies.copy()

        for i in range(self.size):
            reviews_strategy = self.rng.random() < rho[i]

            if reviews_strategy:
                neighbors = list(self.graph.neighbors(i))

                if len(neighbors) > 0:
                    j = self.rng.choice(neighbors)
                    q = self.fermi_probability(i, j)

                    if self.rng.random() < q:
                        new_strategies[i] = self.strategies[j]
                    else:
                        new_strategies[i] = self.strategies[i]

                else:
                    new_strategies[i] = self.strategies[i]

            else:
                new_strategies[i] = int(self.rng.random() < p_social[i])

            # Error de ejecución
            if self.rng.random() < self.config.epsilon_execution:
                new_strategies[i] = 1 - new_strategies[i]

        self.strategies = new_strategies
        self.history.append(self.strategies.copy())

    def _save_metrics(self, round_number: int):
        """
        Guarda métricas agregadas de la comunidad en una ronda concreta.
        """
        self.compute_payoffs()

        self.metrics_history.append({
            "community_id": self.community_id,
            "round": round_number,
            "cooperation_rate": self.cooperation_rate(),
            "mean_payoff": float(np.mean(self.payoffs))
        })

    def run(self, num_rounds: int | None = None):
        """
        Evoluciona la comunidad durante num_rounds rondas.
        """
        if num_rounds is None:
            num_rounds = self.config.Tin

        for round_number in range(1, num_rounds + 1):
            self.step()
            self._save_metrics(round_number=round_number)

    def cooperation_rate(self) -> float:
        """
        Proporción actual de cooperadores.
        """
        return float(np.mean(self.strategies))

    def mean_payoff(self) -> float:
        """
        Payoff medio actual.
        """
        self.compute_payoffs()
        return float(np.mean(self.payoffs))

    def summary(self) -> dict:
        """
        Resumen agregado de la comunidad.
        """
        return {
            "community_id": self.community_id,
            "size": self.size,
            "cooperation_rate": self.cooperation_rate(),
            "mean_payoff": self.mean_payoff()
        }
    
    def get_metrics_history(self):
        """
        Devuelve el historial de métricas de la comunidad.
        """
        return self.metrics_history