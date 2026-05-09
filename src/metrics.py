import pandas as pd


def community_metrics_to_dataframe(community) -> pd.DataFrame:
    """
    Convierte el historial de métricas de una comunidad en un DataFrame.
    """
    return pd.DataFrame(community.get_metrics_history())


def model_metrics_to_dataframe(model) -> pd.DataFrame:
    """
    Convierte el historial de métricas de todas las comunidades en un único DataFrame.
    """
    all_metrics = []

    for community in model.communities:
        all_metrics.extend(community.get_metrics_history())

    return pd.DataFrame(all_metrics)