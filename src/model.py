import numpy as np
import pandas as pd

from src.config import ModelConfig
from src.community import Community
from src.networks import create_intercommunity_network


class MultilevelCooperationModel:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.communities = self._create_communities()

    def _create_communities(self):
        communities = []

        for m in range(self.config.num_communities):
            size = self.rng.integers(
                self.config.min_community_size,
                self.config.max_community_size + 1
            )

            community = Community(
                community_id=m,
                size=int(size),
                config=self.config,
                seed=self.config.seed + m
            )

            communities.append(community)

        return communities

    def run_internal_dynamics(self):
        """
        Ejecuta la dinámica interna de todas las comunidades.
        """
        for community in self.communities:
            community.run()

    def community_summaries(self) -> pd.DataFrame:
        """
        Devuelve un DataFrame con los resultados de cada comunidad.
        """
        summaries = [community.summary() for community in self.communities]
        return pd.DataFrame(summaries)
    
    def get_metrics_history(self):
        """
        Devuelve el historial de métricas de todas las comunidades.
        """
        all_metrics = []

        for community in self.communities:
            all_metrics.extend(community.get_metrics_history())

        return all_metrics
    
    def build_intercommunity_network(self):
        """
        Construye la red entre comunidades a partir de sus tamaños.
        """
        sizes = [community.size for community in self.communities]

        self.intercommunity_graph = create_intercommunity_network(
            sizes=sizes,
            base_prob=self.config.intercommunity_base_prob,
            seed=self.config.seed
        )

        return self.intercommunity_graph
    
    def _normalized_community_payoffs(self, summaries: pd.DataFrame) -> np.ndarray:
        """
        Normaliza los payoffs medios comunitarios al intervalo [0, 1].
        """
        payoffs = summaries["mean_payoff"].to_numpy(dtype=float)

        min_payoff = payoffs.min()
        max_payoff = payoffs.max()

        if max_payoff == min_payoff:
            return np.ones_like(payoffs)

        return (payoffs - min_payoff) / (max_payoff - min_payoff)

    def compute_external_cooperation_probabilities(self) -> pd.DataFrame:
        """
        Calcula la probabilidad de cooperación externa de cada comunidad.

        P_ext = beta_x * x_m + beta_pi * pi_hat_m
        """
        summaries = self.community_summaries().copy()

        x = summaries["cooperation_rate"].to_numpy(dtype=float)
        pi_hat = self._normalized_community_payoffs(summaries)

        p_ext = (
            self.config.beta_x * x
            + self.config.beta_pi * pi_hat
        )

        summaries["normalized_mean_payoff"] = pi_hat
        summaries["external_cooperation_probability"] = np.clip(p_ext, 0.0, 1.0)

        return summaries

    def sample_external_actions(self) -> pd.DataFrame:
        """
        Genera la acción externa de cada comunidad.

        1 = coopera externamente
        0 = defecta externamente
        """
        external_df = self.compute_external_cooperation_probabilities()

        actions = []

        for p in external_df["external_cooperation_probability"]:
            action = int(self.rng.random() < p)
            actions.append(action)

        external_df["external_action"] = actions

        return external_df
    
    def _community_payoff_pair(self, s_i: int, s_j: int) -> float:
        """
        Payoff de una comunidad al interactuar con otra.
        """
        if s_i == 1 and s_j == 1:
            return 1.0
        if s_i == 1 and s_j == 0:
            return 0.0
        if s_i == 0 and s_j == 1:
            return self.config.b_community
        return 0.0

    def run_intercommunity_interactions(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Ejecuta las interacciones entre comunidades conectadas.

        Devuelve:
        - external_df: resumen por comunidad con probabilidad externa, acción y payoff externo.
        - edge_df: resumen por interacción entre comunidades.
        """
        if not hasattr(self, "intercommunity_graph"):
            self.build_intercommunity_network()

        external_df = self.sample_external_actions()

        external_payoffs = np.zeros(len(self.communities))
        edge_results = []

        for i, j in self.intercommunity_graph.edges():
            s_i = int(external_df.loc[external_df["community_id"] == i, "external_action"].iloc[0])
            s_j = int(external_df.loc[external_df["community_id"] == j, "external_action"].iloc[0])

            payoff_i = self._community_payoff_pair(s_i, s_j)
            payoff_j = self._community_payoff_pair(s_j, s_i)

            external_payoffs[i] += payoff_i
            external_payoffs[j] += payoff_j

            edge_results.append({
                "community_i": i,
                "community_j": j,
                "action_i": s_i,
                "action_j": s_j,
                "payoff_i": payoff_i,
                "payoff_j": payoff_j
            })

        external_df["external_payoff"] = external_payoffs

        edge_df = pd.DataFrame(edge_results)

        return external_df, edge_df

    def expected_global_cooperation(self) -> float:
        """
        Cooperación global esperada a partir de las probabilidades externas.
        """
        external_df = self.compute_external_cooperation_probabilities()
        return float(external_df["external_cooperation_probability"].mean())

    def observed_global_cooperation(self, external_df: pd.DataFrame) -> float:
        """
        Cooperación global observada a partir de las acciones externas realizadas.
        """
        return float(external_df["external_action"].mean())