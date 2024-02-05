"""Various approaches to splitting up data into train and test sets (labeling).

If not specified otherwise, X should always be a dataframe with a MultiIndex, which is what
create_meta_features() from meta.py returns and y should be a 2-dimensional DataFrame containing
1s and 0s (or True and False), which is what get_actual_errors() from datasets.py returns.
"""

import math
import torch
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from baseline.setup.shapley.measures import KNN_Shapley
from saged.meta import predict, train_meta_classifiers


def stratified_split(X, y, num_labels=0.20):
    """ Use stratified sampling to split the given data into train and test.

    Parameters:
        X (pandas.core.frame.DataFrame): Input data.
        y (pandas.core.frame.DataFrame): Output data.
        num_labels (int, optional): Number of labeled tuples (training data). Defaults to 20.

    Returns:
        (DataFrame, DataFrame, DataFrame, DataFrame): X_train, X_test, y_train, y_test.
    """
    
    # Initialize the StratifiedShuffleSplit object with the desired number of iterations and test size
    sss = StratifiedShuffleSplit(n_splits=1, test_size=1-num_labels, random_state=42)

    # Split the data into training and testing sets using stratified sampling
    for train_index, test_index in sss.split(X, y):
        # Extract the corresponding subsets of the data
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    return X_train, X_test, y_train, y_test

def random_split(X, y, num_labels=20, random_seed=42):
    """Randomly splits the given data into train and test.

    Parameters:
        X (pandas.core.frame.DataFrame): Input data.
        y (pandas.core.frame.DataFrame): Output data.
        num_labels (int, optional): Number of labeled tuples (training data). Defaults to 20.

    Returns:
        (DataFrame, DataFrame, DataFrame, DataFrame): X_train, X_test, y_train, y_test.
    """
    return train_test_split(X, y, np.arange(len(y)), train_size=num_labels)


def prop_pos_labels(X, y, config, num_labels=20, verbose=False):
    if verbose:
        print(f"* Get {num_labels} initial labels... ", end="", flush=True)

    X_train, X_test, y_train, y_test = random_split(X, y, num_labels)

    if verbose:
        print("done.")

    # Keep a copy of the original X_test
    X_test_copy = X_test.copy()

    while len(X_test) > 0:
        if verbose:
            print(
                "* Train meta classifiers and add positive predictions as labels... "
                , end="", flush=True
            )

        meta_classifiers = train_meta_classifiers(X_train, y_train, config.classifier_type)

        y_pred = predict(meta_classifiers, X_test)

        # Select rows that contain at least one positive prediction
        index = y_pred[(y_pred == 1).any(axis="columns")].index

        # If no positive prediction has been made, we cannot add any more labels
        if len(index) == 0:
            if verbose:
                print("done. (no more positive predictions available)")

            break

        # Otherwise, add values to train set and drop them from test set
        X_train = pd.concat([X_train, X_test.loc[index]])
        y_train = pd.concat([y_train, y_pred.loc[index]])

        X_test = X_test.drop(index)

        if verbose:
            print(f"done. ({len(index)} labels added)")

    return X_train, X_test_copy, y_train, y_test

def clustering_based_sampling(X, y, num_labels=20, propagate=False, verbose=False):
    """Performs the clustering-based sampling described in Raha to label tuples in X with their
    value in y.

    Parameters:
        X (pandas.core.frame.DataFrame): Input data.
        y (pandas.core.frame.DataFrame): Output data.
        propagate (bool, optional): If given, propagate the user labels to create a larger
            training set. Defaults to false.
        num_labels (int, optional): Number of labeled tuples (training data). Defaults to 20.
        verbose (bool): If True, print verbose messages. Defaults to False

    Returns:
        (DataFrame, DataFrame, DataFrame, DataFrame): X_train, X_test, y_train, y_test.
    """

    if num_labels < 2:
        raise ValueError(f"Number of labels must be at least 2 (given: {num_labels}).")

    # Remember which tuples were "labeled" by storing the index of their value in y
    labeled_tuples = []

    for k in range(2, num_labels+2):
        if verbose:
            print(f"* Label tuple {k-1}... ", end="", flush=True)

        # Cluster each column based on the meta features generated for that column
        cluster_models = [
            AgglomerativeClustering(n_clusters=k).fit(X[column]) for column in X.columns.levels[0]
        ]

        # Get cluster labels for each column
        clusters = pd.DataFrame([model.labels_ for model in cluster_models])

        # Count labels per cluster
        labels_per_cluster = np.zeros(k)

        for index in labeled_tuples:
            for cell in clusters[index]:
                labels_per_cluster[cell] += 1

        # Each cell in this dataframe represents the number of labels present in the cluster of
        # the specific cell
        label_count = clusters.applymap(lambda x: labels_per_cluster[x])

        # We get the probability distribution by applying the exp-function to the label count and
        # summing up the columns (see Raha paper)
        p = label_count.applymap(math.exp).sum()

        # Convert to float64 and normalize the distribution
        p = p.astype('float64')
        p /= p.sum()

        # Sample a tuple based on the distribution and "label" it
        while True:
            candidate = np.random.choice(len(p), p=p)
            if candidate not in labeled_tuples:
                labeled_tuples.append(candidate)
                break

        if verbose:
            print("done.")

    # Split X and y into train and test sets
    X_train, X_test = X.iloc[labeled_tuples], X.drop(labeled_tuples)
    y_train, y_test = y.iloc[labeled_tuples], y.drop(labeled_tuples)

    if propagate:
        if verbose:
            print("* Propagate labels... ", end="", flush=True)

        # propagate_labels() takes one column, but we need to propagate the labels for all columns
        propagated_xs = {}
        propagated_ys = {}

        for column in X_train.columns.levels[0]:
            prop_x, prop_y = propagate_labels(X_train[column], X_test[column], y_train[column])
            propagated_xs[column] = prop_x
            propagated_ys[column] = prop_y

        # Concatenation may mess up the datatypes
        X_train = pd.concat(propagated_xs, axis=1).astype(bool)
        y_train = pd.concat(propagated_ys, axis=1).astype(bool)

        if verbose:
            print("done.")

    # Get the test indexes
    X_idx = list(X.index)
    
    return X_train, X_test, y_train, y_test, labeled_tuples, [X_idx[i] for i in range(len(X_idx)) if i not in labeled_tuples]

def propagate_labels(labeled, unlabeled, labels, homogenity=False):
    """Propagates the given labels to generate a larger training dataset.

    labeled and unlabeled should be features for only one column.

    Parameters:
        labeled (pandas.core.frame.DataFrame): Labeled input data
        unlabeled (pandas.core.frame.DataFrame): Unlabeled input data
        labels (pandas.core.series.Series): Labels for the labeled input data
        homogenity (bool): If True, do not propagate labels if there is more than one label in a
            cluster. If False, the classification that has the majority in the given labels will be
            used for propagation. Defaults to False.

    Returns:
        (DataFrame, Series): X_train, y_train
    """
    X = pd.concat([labeled, unlabeled])

    # Num. of clusters = num. of labels + 1 (same as in clustering_based_sampling())
    cluster_model = AgglomerativeClustering(n_clusters=len(labeled)+1)
    cluster_model.fit(X)

    X_train = pd.DataFrame(dtype=bool)
    y_train = pd.Series(dtype=bool)

    for k in range(len(labeled)):
        # Indices of labels in cluster k
        label_indices = np.where(cluster_model.labels_[:len(labeled)] == k)[0]

        # Get number of positive (1) and negative (0) labels in the cluster
        num_pos_labels = len(np.where(labels.iloc[label_indices] == 1)[0])
        num_neg_labels = len(np.where(labels.iloc[label_indices] == 0)[0])

        # If there is only one type of label, propagate this label to all other cells
        if num_pos_labels == 0:
            classification = False
        elif num_neg_labels == 0:
            classification = True
        elif homogenity is False and num_pos_labels != num_neg_labels:
            # If both labels are present, homogenity is False and one label has the majority,
            # use that label
            classification = num_pos_labels > num_neg_labels
        else:
            continue

        new_samples = X.iloc[np.where(cluster_model.labels_ == k)[0]]
        X_train = pd.concat([X_train, new_samples])
        y_train = pd.concat(
            [y_train, pd.Series(classification, index=new_samples.index)]
        )

    return (X_train, y_train)

# TODO parameter for batch size in run_saged.py
def active_learning(X, y, meta_classifier_type, num_labels=20, batch_size=10, verbose=False):
    """Performs the active learning labeling approach as described in the ED2 paper from Mahdavi.

    Differences in the implementation:
    * Hyperparameter optimization is not implemented yet
    * Knowledge sharing is not implemented yet
    * Only Min Certainty is implemented as column selector
    * Cells are picked randomly from the selected column

    Note that, while the number of labels is given in tuples/rows, this number is converted into a
    labeling budget of cells (num_labels * length of rows) as Active Learning is based on labeling
    single cells in columns and not full tuples.

    Parameters:
        X (pandas.core.frame.DataFrame): Input data.
        y (pandas.core.frame.DataFrame): Output data.
        num_labels (int, optional): Number of labeled tuples (training data). Defaults to 20.

    Returns:
        (DataFrame, DataFrame, DataFrame, DataFrame): X_train, X_test, y_train, y_test.
    """
    # Multiply label budget (given in rows) with length of rows, resulting in number of cells 
    num_labels *= y.shape[1]

    # We want to split the given data into train and test sets, therefore we mark labeled cells
    # with a boolean map and split the sets at the end of the labeling process
    labeled_cells = pd.DataFrame(False, index=y.index, columns=y.columns)

    # Initial sample
    if verbose:
        print("* Label initial sample... ", end="", flush=True)

    # TODO batch_size > num_labels?
    labeled_cells.iloc[np.random.choice(labeled_cells.index)] = True

    if verbose:
        print(f"done. ({y.shape[1]} cells labeled)")

    # We may end up labeling all cells of one column, therefore we need to remember which
    # columns should no longer be looked at
    skip = []

    while True:
        # Run Active Learning until the label budget runs out
        if labeled_cells.to_numpy().sum() >= num_labels:
            break

        if verbose:
            print("* Label next batch... ", end="", flush=True)

        # Train classifiers with labeled cells
        classifiers = train_meta_classifiers(X, y.mask(~labeled_cells), meta_classifier_type)

        min_certainty = np.inf
        picked_column = None

        for column in y:
            if column in skip:
                continue

            # The classification model gives probabilities for 1 and for 0...
            predictions = classifiers[column].predict_proba(X[column])

            # ... and the certainty is the larger of the two
            certainties = np.partition(predictions, 1)[:, 1]

            avg_certainty = np.average(certainties)
            if avg_certainty < min_certainty:
                min_certainty = avg_certainty
                picked_column = column

        # Label random cell from picked column
        # Fill label budget (batch size or remaining labels)
        size = min(batch_size, num_labels - labeled_cells.to_numpy().sum())

        # Pick cells that are not already labeled
        # TODO size > num. available cells?
        try:
            cells = np.random.choice(
                labeled_cells.index[~labeled_cells[picked_column]], replace=False, size=size
            )
            labeled_cells[picked_column][cells] = True
        except ValueError:
            # We tried to sample more cells than available
            # (which means that all cells in this column should be labeled)
            labeled_cells[picked_column] = True
            skip.append(picked_column)
            print(skip)

        if verbose:
            print(f"done. ({num_labels - labeled_cells.to_numpy().sum()} labels available)")

    # TODO This works, but is probably not the best split (training data is in test data?)
    return X, X, y.mask(~labeled_cells), y

def knn_shapley(X_train, X_test, y_train, y_test, verbose=False):
    
    # Initialize a KNN-Shapley object
    measure = KNN_Shapley(K=2)
    
    # Define the maximum length of the training data, which is used to filter out the less important tuples 
    max_length = len(X_train)

    # Initialize a tensor to store the filtered indexes
    final_indexes = torch.tensor([])
    
    for col in y_train.columns:
        
        # Convert the dataframes into tensors
        X_train_torch = torch.tensor(X_train[col].values).float()
        y_train_torch = torch.tensor(y_train[col].values).float()
        X_test_torch = torch.tensor(X_test[col].values).float()
        y_test_torch = torch.tensor(y_test[col].values).float()
        
        # Estimate the importance of each tuple in the training data (X_train2_torch)
        tuple_importance = measure._get_shapley_value_torch(X_train_torch, y_train_torch, X_test_torch, y_test_torch)
        # Estimate Shapley values using the meta classifiers (omitted since it requires too much time)
        # tuple_importance = measure.score(X_train2_torch, y_train2_torch, X_test2_torch, y_test2_torch, "NN", meta_classifiers[col])
            
        # Find the largest k values in the result2 tensor 
        # Define the threshold K
        k = int(len(tuple_importance) * 0.2)
        threshold, _ = torch.kthvalue(tuple_importance, len(tuple_importance) - k + 1)

        # Find the indexes of the values greater than or equal to the threshold
        indexes = torch.where(tuple_importance >= threshold)[0]

        # Exclude the results of such columns whose tuple importance is the same for all tuples
        # Consider only the columns whose tuples have different importance 
        if len(indexes) == len(tuple_importance):
            continue
        else:
            final_indexes = indexes if len(indexes) < max_length else final_indexes
            max_length = len(final_indexes)
    
    if final_indexes.size() != 0:
    
        if verbose:
            print("Filtering out the less important tuples in the tranining data...", end="", flush=True)

        # Select the tuples with high importance
        X_train_shapley = X_train.iloc[final_indexes.tolist()]
        y_train_shapley = y_train.iloc[final_indexes.tolist()]
        
        if verbose:
            print("done.")
    
    else:
        # All columns have tuples with equal importance!
        X_train_shapley = X_train
        y_train_shapley = y_train
        
    return X_train_shapley, y_train_shapley