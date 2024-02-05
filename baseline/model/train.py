####################################################################################################################
# Train: Implement a model training method for regression, binary classification, and multi-class classification
# Authors: Mohamed Abdelaal
# Date: June 2022
# Software AG
# All Rights Reserved
####################################################################################################################

import os
import sys
import json
import time
import pandas as pd
import numpy as np
from tensorflow import keras
from datetime import datetime
from keras.backend import clear_session
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, mean_squared_error
from scikeras.wrappers import KerasClassifier, KerasRegressor

from baseline.setup.detectors.detect_method import DATA_PATH, EXP_PATH
from baseline.dataset.preprocess import preprocess
from baseline.dataset.dataset import Dataset
from baseline.model.build_model import build_model
from baseline.model.hyperparams import get_hyperparams
from baseline.model.metrics import evaluate_model
from baseline.model.utils import create_results_path, store_results_csv, plot_learning_curves, ExperimentType, \
    ExperimentName



def train_model(data_df,
                data_name,
                tune_params=False,
                exp_name='',
                exp_type='',
                nb_trails=50,
                epochs=500,
                verbose=True,
                error_rate=0.4,
                nb_generated_samples=0):
    """
    Train a regressor/classifier using Keras
    """

    # Clear clutter from previous keras sessions
    clear_session()

    # Load default hyperparams or tune them
    hyperparams = get_hyperparams(data_df, data_name, nb_trails=nb_trails, epochs=epochs, tune_params=tune_params,
                                  verbose=verbose)

    # Initialize a dictionary for storing the various metrics
    metrics_dict = {}

    # Extract hyperparams
    learning_rate = hyperparams['learning_rate']
    n_hidden = hyperparams['n_hidden']
    n_neurons = hyperparams['n_neurons']

    # Create a Dataset object to retrieve the labels list
    data_obj = Dataset(data_name)
    labels = data_obj.cfg.labels
    ml_task = data_obj.cfg.ml_task

    # Prepare the data
    if verbose:
        print(f"Preprocessing the {data_name} dataset...", end="", flush=True)
        
    X_train_full, y_train_full, X_test, y_test = preprocess(data_df, labels_list=labels, ml_task=ml_task)
    # Extract a validation set
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full)

    # Convert features to array if they are sparse matrices
    X_train = X_train if type(X_train) == np.ndarray else X_train.toarray()
    X_valid = X_valid if type(X_valid) == np.ndarray else X_valid.toarray()
    X_test = X_test if type(X_test) == np.ndarray else X_test.toarray()

    if verbose:
        print("done.")
    
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
        model = KerasRegressor(build_model, input_shape=X_train.shape[1:], learning_rate=learning_rate,
                               n_hidden=n_hidden, n_neurons=n_neurons)

    elif ml_task in ['binary_classification', 'multiclass_classification']:
        nb_classes = 2 if ml_task == 'binary_classification' else y_train.shape[1]
        model = KerasClassifier(build_model, input_shape=X_train.shape[1:], ml_task=ml_task, nb_classes=nb_classes,
                                learning_rate=learning_rate, n_hidden=n_hidden, n_neurons=n_neurons)

    else:
        raise NotImplemented

    # Train the model
    start = time.time()
    model.fit(X_train, y_train, epochs=epochs, validation_data=(X_valid, y_valid),
              callbacks=[early_stopping, model_checkpoint])

    # Plot the learning curve
    # metric_name = 'RMSE' if ml_task == 'regression' else 'Accuracy'
    # plot_learning_curves(model.history_, data_name, experiment_type=exp_type, metric_name=metric_name)

    # Estimate the training time
    metrics_dict.update(training_time=time.time()-start)
    metrics_dict.update(epochs=epochs)
    metrics_dict.update(learning_rate=learning_rate)
    metrics_dict.update(n_hidden=n_hidden)
    metrics_dict.update(n_neurons=n_neurons)
    metrics_dict.update(timestamp=datetime.now())
    metrics_dict.update(nb_generated_samples=nb_generated_samples)
    metrics_dict.update(error_rate=error_rate)

    # Evaluate the models
    prediction = model.predict(X_test)
    # Evaluate the predictive accuracy
    metrics_dict = evaluate_model(metrics_dict, y_test, prediction, ml_task, verbose=True)

    # Prepare a path to store the results
    filename = exp_type + '_' + data_name + '.csv'
    results_path = create_results_path(data_name, exp_name, filename)

    # Store the results
    store_results_csv(metrics_dict, results_path)


if __name__ == '__main__':
    # Get the data path
    dataset_name = 'housing'

    dataset_path = os.path.abspath(os.path.join(DATA_PATH, dataset_name))
    # Retrieve the dirty and clean data
    data_path = os.path.abspath(os.path.join(DATA_PATH, dataset_name, 'clean.csv'))

    # Load the data
    dataset_df = pd.read_csv(data_path, header="infer", encoding="utf-8", low_memory=False)

    # Train a model
    experiment_name = ExperimentName.MODELING.__str__()
    experiment_type = ExperimentType.GROUND_TRUTH.__str__()
    train_model(dataset_df,
                dataset_name,
                tune_params=False,
                exp_name=experiment_name,
                exp_type=experiment_type,
                verbose=True,
                epochs=10)

    # experiment_type: baseline, ground_truth, other baselines
