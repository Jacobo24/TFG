import numpy as np
import pandas as pd

from src.config import ModelConfig
from src.community import Community


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