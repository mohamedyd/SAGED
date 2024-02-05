"""Create meta features and train meta classifiers."""

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from saged.classification import initialize_classifier
from saged.featurization import create_features
from saged.similarity import get_similarity

def create_meta_features(dirty_dataset, historical_datasets, config, verbose=False):
    """Creates meta features (outputs of classifiers from historical datasets) for the dirty
    dataset. The classifier models for the historical datasets must already be trained.

    Under the hood, this method performs the following steps:
        1. Calculate the similarity of dirty columns to columns of historical datasets
        2. Create features of the dirty dataset (see create_features()).
        3. For each dirty column, get outputs of classifiers from similar historical columns.

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
            the second index is the name of a historical dataset, the third index is the column
            from the historical dataset.
    """
    meta_features = {}

    if verbose:
        print("* Create features for dirty dataset... ", end="", flush=True)

    # Create features for dirty dataset
    all_features = create_features(dirty_dataset.dirty_df)

    if verbose:
        print("done")

    # We need to get datasets by name later
    hist_datasets_by_name = {dataset.name: dataset for dataset in historical_datasets}

    similarity = get_similarity(dirty_dataset, historical_datasets, config, verbose=verbose)

    # Iterate over columns in dirty dataset and get their clusters
    for column in dirty_dataset.dirty_df:
        if verbose:
            print(f"* Create meta features for column '{column}'... ", end="", flush=True)

        # Get features for this column
        features = all_features[column]

        # Iterate over similar columns to dirty column
        for name, attributes in similarity[column].items():
            hist_dataset = hist_datasets_by_name[name]
            for attribute in attributes:
                classifier = hist_dataset.load_classifier_model(
                    config.classifier_type, attribute
                )

                # TODO How to deal with NaN?
                meta_features[(column, name, attribute)] = classifier.predict(features.fillna(0))
        
        if verbose:
            print("done.")

    return pd.DataFrame(meta_features, dtype=int)

def train_meta_classifiers(X, y, meta_classifier_type, verbose=False):
    """Trains and evaluates meta classifiers of the given type for every column in y.
    The X dataframe must have a MultiIndex with the first element being the respective column to
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
                f"* Train (meta) classifier ('{meta_classifier_type.value}') for '{column}'... ",
                end="", flush=True
            )

        # Drop values from X and y that are "unlabeled" (set to NaN)
        X_train = X[column][y[column].notna()]
        y_train = y[column].dropna()

        classifier = initialize_classifier(meta_classifier_type)
        classifier.fit(X_train, y_train)
        classifiers[column] = classifier

        if verbose:
            print("done.")

    return classifiers

def predict(meta_classifiers, X_test):
    predictions = []

    for column in X_test.columns.levels[0]:
        predictions.append(meta_classifiers[column].predict(X_test[column]))

    return pd.DataFrame(
        predictions, index=X_test.columns.levels[0], columns=X_test.index
    ).T

def evaluate(meta_classifiers: dict = {}, 
             X_test: pd.DataFrame = "", 
             y_test: pd.DataFrame = "", 
             y_train: pd.DataFrame = "", 
             train_indices: list = [], 
             test_indices: list = []):
    """Evaluate the meta classifiers on each column of the test set and return one set of
    precision, recall and F1-score values.

    Parameters:
        meta_classifiers (dict): A dictionary mapping each column name on the respective classifier.
        X_test (pandas.core.frame.DataFrame): Meta features (see create_meta_features()).
        y_test (pandas.core.frame.DataFrame): Binary classification of errors
        test_indices (list): indices of the test features relative to the original dirty dataset

    Returns:
        (float, float, float): Precision, recall and F1-score.
    """

    y_true = []
    y_pred_df = pd.DataFrame()
    y_pred_list = []
    detection_dict = {}

    for j, column in enumerate(y_test.columns):
        y_true.extend(y_test[column])
        y_pred_col = meta_classifiers[column].predict(X_test[column])
        y_pred_df[column] = y_pred_col
        # Generate the detection dictionary
        for i, label in enumerate(y_pred_col):
            if label:
                detection_dict[(test_indices[i],j)] = 'JUST A DUMMY VALUE'
        y_pred_list.extend(y_pred_col)
     
    # Adjust the indexes of the y_pred_df
    y_pred_df = y_pred_df.set_index(y_test.index) 
     
    # Add the dirty instances in the training data to the detection dict
    for j, column in enumerate(y_train.columns):
        for i, label in enumerate(y_train[column]):
            try:
                if label:
                    detection_dict[(train_indices[i],j)] = 'JUST A DUMMY VALUE'
            except:
                print("Index not found!")
                continue
    
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        y_true, y_pred_list, average="binary", zero_division=0)

    return precision, recall, f1_score, detection_dict, y_pred_df
