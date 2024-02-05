############################################################################
# Benchmark: Build a model which can be used for classification or regression
# Authors: Mohamed Abdelaal
# Date: June 2022
# Software AG
# All Rights Reserved
#############################################################################

from tensorflow import keras
from keras.metrics import MeanSquaredError, CategoricalAccuracy, BinaryAccuracy, Accuracy


def build_model(learning_rate, n_hidden, n_neurons, input_shape=[5], ml_task='regression', nb_classes=2):
    """
     Build and compile a Keras model
    """

    # Define a sequential model
    model = keras.models.Sequential()
    options = {"input_shape": input_shape}

    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation="relu", **options))
        options = {}

    # Add an output layer and the optimizer
    optimizer = keras.optimizers.Adam(learning_rate)

    if ml_task == "multiclass_classification":
        print("nb_classes", nb_classes)
        model.add(keras.layers.Dense(nb_classes, activation="softmax"))
        model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=CategoricalAccuracy())

    elif ml_task == "binary_classification":
        model.add(keras.layers.Dense(1, activation="sigmoid"))
        model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=[BinaryAccuracy(), Accuracy()])

    elif ml_task == 'regression':
        model.add(keras.layers.Dense(1))
        model.compile(optimizer=optimizer, loss="mean_squared_error", metrics=MeanSquaredError())

    else:
        raise NotImplemented

    return model
