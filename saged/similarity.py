"""Methods to measure the similarity of dataset columns."""

from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from saged.cluster import initialize_cluster_model
from saged.configuration import Similarity
from saged.profiler import get_profiles

def clustering(dirty_dataset, historical_datasets, config, verbose=False):
    """Create column profiles for given datasets and cluster the columns based on those profiles.
    The output is a dictionary where the keys are columns of the dirty dataset and the
    corresponding values are, again, dictionaries specifying which columns of the historical
    datasets are similar to the dirty dataset. The keys of the nested dictionaries give the
    historical dataset's name, the value is a list of similar columns from the historical dataset.

    Parameters:
        dirty_dataset (saged.datasets.Dataset): Dirty dataset.
        historical_datasets (list[saged.datasets.Dataset]): Historical datasets.
        config(saged.configuration.Configuration): Configuration with profile type, cluster
            algorithm and number of clusters
        verbose (bool, optional): If True, print verbose messages. Defaults to False.

    Return:
        dict: The dirty columns mapped to similar columns from the historical datasets
    """
    dirty_profiles, historical_profiles = get_profiles(
        dirty_dataset, historical_datasets, config, verbose=verbose
    )

    if verbose:
        print("* Cluster columns in historical datasets... ", end="", flush=True)

    cluster_model = initialize_cluster_model(
        config.cluster_algorithm, config.n_clusters
    )

    cluster_model.fit(historical_profiles.T)

    # Get columns by cluster
    cluster_map = pd.DataFrame(cluster_model.labels_, index=historical_profiles.columns, dtype=int)
    clusters = [defaultdict(list) for _ in range(config.n_clusters)]

    for index, label in cluster_map.iterrows():
        # The index is a tuples consisting of the dataset and column name
        clusters[label[0]][index[0]].append(index[1])

    if verbose:
        print("done.")

    similarity = {}

    for column, label in zip(dirty_profiles, cluster_model.predict(dirty_profiles.T)):
        similarity[column] = clusters[label]

    return similarity

def __multiindex_to_dict(multiindex):
    """Convert a MultiIndex with two levels to a defaultdict.

    Args:
        multiindex (pandas.MultiIndex): a MultiIndex with two levels.

    Returns:
        defaultdict(list): MulitIndex converted to defaultdict.
    """
    dictionary = defaultdict(list)
    for key, value in multiindex:
        dictionary[key].append(value)

    return dictionary

def __metric_based(dirty_dataset, historical_datasets, config, metric, verbose=False):
    """A general metric-based similarity measure.
    The output has the same format as in clustering(), see above.

    Parameters:
        dirty_dataset (saged.dataset.Dataset): Dirty dataset.
        historical_datasets (list[saged.dataset.Dataset]): Historical datasets.
        config (saged.configuaration.Configuration): SAGED Configuration.
        metric (function): Some metric mapping two array-like variables two a real number.
        verbose (bool, optional): If True, print verbose messages. Defaults to False.

    Returns:
        dict: The dirty columns mapped to similar columns from the historical datasets.
    """
    dirty_profiles, hist_profiles = get_profiles(
        dirty_dataset, historical_datasets, config, verbose=verbose
    )

    if verbose:
        print(
            "* Measure the similarity of dirty columns and historical columns... ", 
            end="", flush=True
        )

    similarity = {}

    # TODO This could be done in get_similarity (save time of profiling)
    # If n_meta_features is 0 (default value) or larger than the number of historical columns, use
    # all historical columns to generate meta features
    if config.n_meta_features == 0 or config.n_meta_features >= hist_profiles.shape[1]:
        columns = __multiindex_to_dict(hist_profiles.columns)
        for column in dirty_profiles:
            similarity[column] = columns

        if verbose:
            print("done.")

        return similarity

    # Otherwise generate n_meta_features meta features for each column

     
    for column in dirty_profiles:
        
        # Initialize a distances list
        distances_list = []
        
        # Normalize the vectors
        norm1 = np.linalg.norm(dirty_profiles[column])
        normalized_vector1 = dirty_profiles[column] / norm1
            
        # Calculate distance of dirty column to every historical column
        for hist_column in hist_profiles:
            
            # Normalize the vectors
            norm2 = np.linalg.norm(hist_profiles[hist_column])
            normalized_vector2 = hist_profiles[hist_column] / norm2
            
            distances_list.append(metric(normalized_vector1, normalized_vector2))
        
        distances = np.array(distances_list)
        
        #distances = np.array([
        #    metric(dirty_profiles[column], hist_profiles[hist_column])
        #    for hist_column in hist_profiles])

        # Add columns with the lowest distances
        similarity[column] = __multiindex_to_dict(
            hist_profiles.columns[distances.argsort()[:config.n_meta_features]])

    if verbose:
        print("done.")

    return similarity

def get_similarity(dirty_dataset, historical_datasets, config, verbose=False):
    """Calculate the similarity of the dirty columns and the historical columns with the similarity
    measure specified in the configuration.

    Parameters:
        dirty_dataset (saged.datasets.Dataset): Dirty dataset.
        historical_datasets (list[saged.datasets.Dataset]): Historical datasets.
        config(saged.configuration.Configuration): Configuration with profile type, cluster
            algorithm and number of clusters
        verbose (bool, optional): If True, print verbose messages. Defaults to False.

    Returns:
        dict: The dirty columns mapped to similar columns from the historical datasets.
    """
    if config.similarity is Similarity.CLUSTERING:
        return clustering(dirty_dataset, historical_datasets, config, verbose=verbose)
    if config.similarity is Similarity.COSINE:
        return __metric_based(dirty_dataset, historical_datasets, config, cosine, verbose=verbose)
