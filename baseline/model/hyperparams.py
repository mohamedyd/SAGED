#####################################################################
# Hyperparams: Implement an Optuna-based hyperparameter tuning method
# Authors: Mohamed Abdelaal
# Date: June 2022
# Software AG
# All Rights Reserved
#####################################################################

import os
import json
import sys
import numpy as np
import pandas as pd
import optuna
from baseline.dataset.dataset import Dataset
from baseline.dataset.preprocess import preprocess
from keras.backend import clear_session
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from baseline.model.build_model import build_model
from scikeras.wrappers import KerasClassifier, KerasRegressor
from baseline.setup.detectors.detect_method import DetectMethod, EXP_PATH, DATA_PATH


def get_hyperparams(data_df, dataset_name, nb_trails, epochs, tune_params=False, verbose=True):
    """
    Load the default hyperparams or tune them. The action is based on the predicate tune_params
    """
    # Initialize a dict to store the hyperparams
    hyperparams = {}

    if tune_params:
        # Check if the tuned parameters already exist
        tuned_path = os.path.join(EXP_PATH, 'hyperparams', dataset_name, 'tuned.json')
        if os.path.exists(tuned_path):
            # Load the JSON file
            with open(tuned_path, 'r') as json_file:
                hyperparams = json.load(json_file)
                if hyperparams:
                    if verbose:
                        print("Hyperparameters already exist for the {} dataset".format(dataset_name))
                    return hyperparams
                else:
                    # Remove the JSON file if it is empty, before running the hyperparams tuning method
                    os.remove(tuned_path)
        else:
            if verbose:
                print("Optuna: Start hyperparameters tuning ..")
    else:
        # Load the default parameters if tuning is not required
        if verbose:
            print("Loading the default hyperparameters...", end="", flush=True)
        default_path = os.path.join(EXP_PATH, 'hyperparams', 'default.json')
        with open(default_path, 'r') as json_file:
            if verbose:
                print("done.")
            return json.load(json_file)

    # Execute Optuna to tune the hyperparameters
    hyperparams = tune_hyperparams(data_df, dataset_name, nb_trails=nb_trails, epochs=epochs, verbose=verbose)

    return hyperparams


def tune_hyperparams(data_df, dataset_name, nb_trails=50, epochs=100, verbose=True):
    """
    Use Optuna to tune the hyperparameters of a model
    """

    # Check if the tuned parameters already exist
    dataset_eval_path = os.path.join(EXP_PATH, 'hyperparams', dataset_name)
    tuned_path = os.path.join(dataset_eval_path, 'tuned.json')
    # Create the dataset folder, if it does not exist
    if not os.path.exists(dataset_eval_path):
        os.mkdir(dataset_eval_path)

    # Load the default hyperparameters
    default_path = os.path.join(EXP_PATH, 'hyperparams', 'default.json')
    with open(default_path, 'r') as json_file:
        default_params = json.load(json_file)

    # Create a dictionary of the hyperparamters to be examined
    param_distributions = {
        'learning_rate': optuna.distributions.LogUniformDistribution(low=1e-5, high=1e-1),
        "n_hidden": optuna.distributions.IntUniformDistribution(low=1, high=4, step=1),
        "n_neurons": optuna.distributions.IntUniformDistribution(low=5, high=100, step=5),
    }

    # Create a Dataset object to retrieve the labels list
    data_obj = Dataset(dataset_name)
    labels = data_obj.cfg.labels
    ml_task = data_obj.cfg.ml_task

    # Prepare the data
    X_train_full, y_train_full, X_test, y_test = preprocess(data_df, labels_list=labels, ml_task=ml_task)
    # Extract a validation set
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full)

    # Convert features to array if they are sparse matrices
    X_train = X_train if type(X_train) == np.ndarray else X_train.toarray()
    X_valid = X_valid if type(X_valid) == np.ndarray else X_valid.toarray()

    # all_features, all_inputs, train_ds, val_ds, test_ds = prepare_data(data_df, labels)

    # Define the batch size
    batch_size = min(200, X_train.shape[0])

    # set callbacks
    early_stopping = EarlyStopping(monitor="val_loss", mode="min", verbose=verbose, patience=10)
    model_checkpoint = ModelCheckpoint(
        "best_model.h5",
        monitor="val_loss",
        mode="min",
        verbose=verbose,
        save_best_only=True,
    )

    # Create a wrapper around the Keras model
    if ml_task == 'regression':
        model = KerasRegressor(build_model, input_shape=X_train.shape[1:], learning_rate=default_params[
            'learning_rate'], n_hidden=default_params['n_hidden'], n_neurons=default_params['n_neurons'])

    elif ml_task in ['binary_classification', 'multiclass_classification']:
        nb_classes = 2 if ml_task == 'binary_classification' else y_train.shape[1]
        model = KerasClassifier(build_model, input_shape=X_train.shape[1:], ml_task=ml_task, nb_classes=nb_classes,
                                learning_rate=default_params['learning_rate'], n_hidden=default_params['n_hidden'],
                                n_neurons=default_params['n_neurons'])

    else:
        raise NotImplemented

    optuna_search = optuna.integration.OptunaSearchCV(
        model, param_distributions, n_trials=nb_trails, timeout=600, verbose=verbose)

    # Start the search process
    optuna_search.fit(X_train, y_train, epochs=epochs, validation_data=(X_valid, y_valid),
                      callbacks=[early_stopping, model_checkpoint])

    # Store the best trial
    # Get the best trial
    trial = optuna_search.study_.best_trial
    with open(tuned_path, 'w') as fp:
        json.dump(trial.params, fp)

    if verbose:
        print("Best trial:")
        print("  Value: ", trial.value)
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

    return trial.params


if __name__ == '__main__':
    # Get the data path
    dataset_name = 'nasa'
    dataset_path = os.path.abspath(os.path.join(DATA_PATH, dataset_name))
    # Retrieve the dirty and clean data
    data_path = os.path.abspath(os.path.join(DATA_PATH, dataset_name, 'clean.csv'))
    # Load the data
    data_df = pd.read_csv(data_path, header="infer", encoding="utf-8", low_memory=False)
    print(EXP_PATH)
    # Train a model
    get_hyperparams(data_df, dataset_name, nb_trails=50, epochs=100, tune_params=True, verbose=False)

