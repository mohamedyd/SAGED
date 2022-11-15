"""Create meta features and train meta classifiers."""

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from saged.classification import initialize_classifier
from saged.cluster import cluster_columns
from saged.profiler import create_column_profiles
from saged.featurization import create_features


def create_meta_features(dirty_dataset, historical_datasets, config, verbose=False):
    """Creates meta features (outputs of classifiers from historical datasets) for the dirty
    dataset. The classifier models for the historical datasets must already be trained.

    Under the hood, this method performs the following steps:
        1. Create column profiles for both the dirty and the historical datasets.
        2. Cluster the historical datasets based on the column profile.
        3. Create features of the dirty dataset (see create_features()).
        4. Get outputs of classifiers from historical datasets (must be trained beforehand).

    The outputs in step 4 are the meta features.

    Parameters:
        dirty_dataset (saged.datasets.Dataset): A dirty dataset.
        historical_datasets (list[saged.datasets<.Dataset]): A list of historical datasets.
        config (saged.configuration.Configuration): A configuration containing profile type,
            cluster algorithm, etc.
        verbose (bool, optional): If True, print verbose status messages. Defaults to False.

    Returns:
        pandas.core.frame.DataFrame: A pandas DataFrame containing the generated meta features with
            a MultiIndex. The first index is the column for which the meta features were generated,
            the second index is a tuple containing the historical dataset name and the name of the
            column whose classifier was used.
    """
    meta_features = {}

    if verbose:
        print("* Create column profiles... ", end="", flush=True)

    column_profiles_dirty = create_column_profiles([dirty_dataset], config.profile_type, False)
    column_profiles_hist = create_column_profiles(historical_datasets, config.profile_type)

    if verbose:
        print("done.")

    if verbose:
        print("* Cluster columns in historical datasets... ", end="", flush=True)

    clusters = cluster_columns(
        column_profiles_dirty.join(column_profiles_hist),
        config.cluster_algorithm,
        config.n_clusters
    )

    if verbose:
        print("done.")

    if verbose:
        print("* Create features for dirty dataset...", end="", flush=True)

    # Create features for dirty dataset
    all_features = create_features(dirty_dataset.dirty_df)

    if verbose:
        print("done")

    # We need to get datasets by name later
    hist_datasets_by_name = {dataset.name: dataset for dataset in historical_datasets}

    # Iterate over columns in dirty dataset and get their clusters
    for column in dirty_dataset.dirty_df:
        cluster = clusters[column]
        if verbose:
            print(f"* Create meta features for column '{column}'... ", end="", flush=True)

        # Get features for this column
        features = all_features[column]

        # Iterate over columns in historical datasets
        for hist_column in column_profiles_hist:
            if clusters[hist_column] == cluster:
                # The column names in the column profile have the format
                # (<dataset-name>, <attr-name>)
                name, attribute = hist_column
                hist_dataset = hist_datasets_by_name[name]

                classifier = hist_dataset.load_classifier_model(
                    config.classifier_type, attribute
                )

                # TODO How to deal with NaN?
                meta_features[(column, hist_column)] = classifier.predict(features.fillna(0))
        if verbose:
            print("done.")

    return pd.DataFrame(meta_features, dtype=int)

def train_meta_classifiers(X, y, meta_classifier_type, verbose=False):
    """Trains and evaluates meta classifiers of the given type for every column in y.
    The X dataframe should have a MultiIndex with the first element being the respective column to
    the outputs of y. y may contain NaN values, which are considered "unlabeled" and will be
    dropped (along with the respective row from X).

    Parameters:
        X (pandas.core.frame.DataFrame): Input data (meta features).
        y (pandas.core.frame.DataFrame): Output data (classified errors, 0 or 1).
        meta_classifier_type (saged.configuration.ClassifierType): Type of meta classifier to train.
        verbose (bool): Print verbose messages if True. Defaults to False.

    Returns:
        dict: A dictionary mapping the dataset's columns to their meta classifiers.
    """

    classifiers = {}
    for column in X.columns.levels[0]:
        if verbose:
            print(
                f"* Train meta classifier ('{meta_classifier_type.value}') for '{column}'... ",
                end="", flush=True
            )

        classifier = initialize_classifier(meta_classifier_type)
        # Drop values from X and y that are "unlabeled" (set to NaN)
        classifier.fit(X[column][y[column].notna()], y[column].dropna())
        classifiers[column] = classifier

        if verbose:
            print("done.")

    return classifiers

def evaluate(meta_classifiers, X_test, y_test):
    """Evaluate the meta classifiers on each column of the test set and return one set of
    precision, recall and F1-score values.

    Parameters:
        meta_classifiers (dict): A dictionary mapping each column name on the respective classifier.
        X_test (pandas.core.frame.DataFrame): Meta features (see create_meta_features()).
        y_test (pandas.core.frame.DataFrame): Binary classification of errors

    Returns:
        (float, float, float): Precision, recall and F1-score.
    """

    y_true = []
    y_pred = []

    for column in y_test.columns:
        y_true.extend(y_test[column])
        y_pred.extend(meta_classifiers[column].predict(X_test[column]))

    precision, recall, f1_score, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    return precision, recall, f1_score
