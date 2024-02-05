#####################################################################################
# Benchmark: Implement data preprocessing methods based on Keras preprocessing layers
# Authors: Mohamed Abdelaal
# Date: June 2022
# Software AG
# All Rights Reserved
#####################################################################################

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from baseline.setup.detectors.detect_method import DetectMethod, EXP_PATH, DATA_PATH
from tensorflow.keras import layers


def separate_categorical(features):
    """
    Separate numerical and categorical attributes
    @arguments
     features -- dataframe, features of a dataset
    @return:
     features_cat -- dataframe, categorical attributes
     features_num -- dataframe, numerical attributes
    """
    # Select the columns whose data type is object
    features_cat = features.select_dtypes(include=['object']).copy()

    # Extract the numerical features
    if not features_cat.empty:
        features_num = features.select_dtypes(exclude=['object']).copy()
    else:
        features_num = features.copy()  # there are no categorical features

    return features_cat, features_num


def df_to_dataset(dataframe, labels_list, shuffle=True, batch_size=32):
    """
     Convert a DataFrame into a tf.data.Dataset, then shuffles and batches the data.
    """
    df = dataframe.copy()
    labels = df.pop(labels_list)

    df = {key: value[:, tf.newaxis] for key, value in dataframe.items()}
    ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))

    ds = ds.batch(batch_size)
    ds = ds.prefetch(batch_size)

    return ds


def get_normalization_layer(name, dataset):
    """
    Generate a layer which applies feature-wise normalization to numerical features

    @arguments:
    name -- string, name of the numerical feature to be prepared
    dataset -- tf.data.Dataset object, input dataset
    """
    # Create a Normalization layer for the feature.
    normalizer = layers.Normalization(axis=None)

    # Prepare a Dataset that only yields the feature.
    feature_ds = dataset.map(lambda x, y: x[name])

    # Learn the statistics of the data.
    normalizer.adapt(feature_ds)

    return normalizer


def get_category_encoding_layer(name, dataset, dtype, max_tokens=None):
    """
    Generate a layer which maps values from a vocabulary to integer indices and multi-hot encodes the features
    """
    # Create a layer that turns strings into integer indices.
    if dtype == 'string':
        index = layers.StringLookup(max_tokens=max_tokens)
    # Otherwise, create a layer that turns integer categorical values into integer indices.
    else:
        index = layers.IntegerLookup(max_tokens=max_tokens)

    # Prepare a `tf.data.Dataset` that only yields the feature.
    feature_ds = dataset.map(lambda x, y: x[name])

    # Learn the set of possible values and assign them a fixed integer index.
    index.adapt(feature_ds)

    # Encode the integer indices.
    encoder = layers.CategoryEncoding(num_tokens=index.vocabulary_size())

    # Apply multi-hot encoding to the indices. The lambda function captures the
    # layer, so you can use them, or include them in the Keras Functional model later.
    return lambda feature: encoder(index(feature))


def prepare_data(dataframe, labels_list, batch_size=32, verbose=True):
    """
    Implement a data preparation pipeline with Keras preprocessing layers
    """

    # Split the DataFrame into training, validation, and test sets
    train, val, test = np.split(dataframe.sample(frac=1), [int(0.8 * len(dataframe)), int(0.9 * len(dataframe))])

    if verbose:
        print(len(train), 'training examples')
        print(len(val), 'validation examples')
        print(len(test), 'test examples')

    # Get a list of headers for numerical and categorical features
    # Split numerical and categorical attributes of the train set
    train_cat, train_num = separate_categorical(train)
    num_attribs = list(train_num)
    cat_attribs = list(train_cat)

    # Generate Tensorflow Dataset objects
    train_ds = df_to_dataset(train, labels_list=labels_list, batch_size=batch_size)
    val_ds = df_to_dataset(val, labels_list=labels_list, shuffle=False, batch_size=batch_size)
    test_ds = df_to_dataset(test, labels_list=labels_list, shuffle=False, batch_size=batch_size)

    # Normalize the numerical features and add them to one list of inputs called encoded_features
    all_inputs = []
    encoded_features = []

    # Numerical features.
    for header in num_attribs:
        numeric_col = tf.keras.Input(shape=(1,), name=header)
        normalization_layer = get_normalization_layer(header, train_ds)
        encoded_numeric_col = normalization_layer(numeric_col)
        all_inputs.append(numeric_col)
        encoded_features.append(encoded_numeric_col)

    # Turn the integer categorical values from the dataset into integer indices, perform multi-hot encoding, and
    # add the resulting feature inputs to encoded_features
    for header in cat_attribs:
        categorical_col = tf.keras.Input(shape=(1,), name=header, dtype='string')
        encoding_layer = get_category_encoding_layer(name=header,
                                                     dataset=train_ds,
                                                     dtype='string',
                                                     max_tokens=5)
        encoded_categorical_col = encoding_layer(categorical_col)
        all_inputs.append(categorical_col)
        encoded_features.append(encoded_categorical_col)

    all_features = tf.keras.layers.concatenate(encoded_features)

    return all_features, all_inputs, train_ds, val_ds, test_ds


if __name__ == '__main__':
    # Get the data path
    dataset_name = 'adult'
    dataset_path = os.path.abspath(os.path.join(DATA_PATH, dataset_name))
    # Retrieve the dirty and clean data
    data_path = os.path.abspath(os.path.join(DATA_PATH, dataset_name, 'dirty.csv'))

    data_df = pd.read_csv(data_path, header="infer", encoding="utf-8", low_memory=False)

    # Train a model
    prepare_data(dataframe=data_df, verbose=True)