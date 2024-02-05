"""Cluster methods."""

from sklearn.cluster import KMeans
from saged.configuration import ClusterAlgorithm

def initialize_cluster_model(cluster_algorithm, n_clusters):
    """Initializes a model the given cluster algorithm with the given number of clusters.

    Parameters:
        cluster_algorithm (saged.configuration.ClusterAlgorithm): A cluster algorithm.
        n_clusters (int): Number of clusters.

    Returns:
        General cluster algorithm: The initialized cluster model.
    """
    if cluster_algorithm is ClusterAlgorithm.KMEANS:
        return KMeans(n_clusters=n_clusters)

    raise ValueError(f"Cluster algorithm {cluster_algorithm} not found.")
