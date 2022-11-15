"""Cluster methods."""

from sklearn.cluster import AgglomerativeClustering, KMeans
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
    if cluster_algorithm is ClusterAlgorithm.WARD_AGGLOMERATIVE:
        return AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")

def cluster_columns(df, cluster_algorithm, n_clusters):
    """Clusters the columns of the given dataframe with the given cluster algorithm into the
    number of clusters specified.

    Parameters:
        df (pandas.core.frame.DataFrame): A pandas DataFrame.
        cluster_algorithm (saged.configuration.ClusterAlgorithm): A cluster algorithm.
        n_clusters (int): Number of clusters.

    Returns:
        dict: A dictionary mapping the columns' names to their cluster.
    """
    cluster_model = initialize_cluster_model(cluster_algorithm, n_clusters)
    cluster_model.fit(df.T)

    labels = cluster_model.labels_.copy()
    labels.shape = (len(labels),1)

    return dict(zip(df.columns, labels.T.tolist()[0]))
