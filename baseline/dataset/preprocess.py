####################################################
# Preprocess: Implement data preprocessing methods
# Authors: Mohamed Abdelaal
# Date: June 2022
# Software AG
# All Rights Reserved
###################################################

import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import logging
import numpy as np
import sys


def split_train_test(data_df, rand_seed):
    try:
        return train_test_split(data_df, test_size=0.2, random_state=rand_seed)
    except:
        logging.info("Not enough data samples to split")


def separate_features(dataset, label_list):
    """
    Separate features and labels
    @arguments
     dataset -- dataframe, the input dataset
     label_list -- list, list of strings denoting the labels
    :return:
    labels -- dataframe, label attributes
    features -- dataframe, feature attributes
    """
    labels = dataset[label_list].copy()
    features = dataset.drop(label_list, axis=1)

    return features, labels


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


def preprocess(dataset, labels_list, ml_task='binary_classification'):
    """
      Prepare a dataset via data separation, scaling, encoding

      @arguments:
        dataset -- dataframe, dataset to be preprocessed
        label_list -- list, list of strings denoting the label attributes
      @return:
        X_train, y_train -- training data
        X_test, y_test --  test data
      """

    # Remove NaN rows
    dataset = dataset.dropna()

    # Separate features and labels
    features, labels = separate_features(dataset, labels_list)

    # Split the dataset into train (80%) and test data (20%)
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, random_state=42,
                                                                                test_size=0.2)

    # Find attributes with mixed types and convert them into str
    for column in train_features.columns:
        data_types = train_features[column].dtype
        unique_data_types = train_features[column].apply(type).unique()

        if len(unique_data_types) > 1:
            print(f"The attribute {column} contains mixed types.")
            train_features[column] = train_features[column].astype(str)
            test_features[column] = test_features[column].astype(str)
        else:
            print(f"The attribute {column} does not have mixed types.")

    # ==== Preparing the numerical and categorical attributes
    # Prepare the numerical pipeline
    num_pipeline = Pipeline([('std_scaler', StandardScaler())])

    # Split numerical and categorical attributes of the train set
    train_cat, train_num = separate_categorical(train_features)
    
    num_attribs = list(train_num)
    cat_attribs = list(train_cat)

    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(handle_unknown='ignore'), cat_attribs)])

    X_train = full_pipeline.fit_transform(train_features)
    X_test = full_pipeline.transform(test_features)

    # Prepare the labels, if necessary
    if not train_labels.empty and ml_task == 'binary_classification':
        encoder = LabelEncoder()
        y_train = encoder.fit_transform(train_labels)
        y_test = encoder.fit_transform(test_labels)
        print(set(y_test))
        print(set(y_train))
        print(set(test_labels))
        print(set(train_labels))

    elif ml_task == 'regression':
        # Normalize the labels
        # y_train = train_labels/train_labels.max()
        # y_test = test_labels/test_labels.max()
        y_train = train_labels
        y_test = test_labels

    elif ml_task == 'multiclass_classification':
        encoder = OneHotEncoder(handle_unknown='ignore')
        y_train = encoder.fit_transform(train_labels.values.reshape(-1, 1)).toarray()
        y_test = encoder.transform(test_labels.values.reshape(-1, 1)).toarray()
    else:
        raise NotImplemented

    return X_train, y_train, X_test, y_test
