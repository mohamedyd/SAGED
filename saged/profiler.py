"""Create profiles for datasets."""

import re
import string
import numpy as np
import pandas as pd

from saged.configuration import ProfileType

def create_character_distributions(df):
    """Creates the character distributions for the given dataset: for each column, every
    character in the column is mapped to the probability that it occurs in a cell.

    Parameters:
        df (pandas.core.frame.DataFrame): A pandas DataFrame.

    Returns:
        pandas.core.frame.DataFrame: The resulting distributions as a pandas DataFrame.
    """
    distributions = {}
    for column in df:
        chars = {}
        for value in df[column]:
            for c in set(list(str(value))):
                chars[c] = 1 if not c in chars else chars[c] + 1

        distributions[column] = {
            c: count / df.shape[0] for c, count in chars.items()
        }

    return pd.DataFrame(distributions).fillna(0)

def create_structure_features(df):
    """Creates a profile based on the structure features of the given dataset, that is:
        1. Fraction of unique cell values
        2. Fraction of explicitly missing cell values
        3. Fraction of alphabetical values
        4. Fraction of numerical cell values
        5. Fraction of punctuation cell values
        6. Fraction of miscellaneous cell values

    Parameters:
        df (pandas.core.frame.DataFrame): A pandas DataFrame.

    Return
        pandas.core.frame.DataFrame: A pandas DataFrame containing the structure features.
    """
    features = np.zeros(6)
    num_observations = len(df)

    for index, column in enumerate(df.columns):
        # Fraction of unique cell values
        value_counts = df[column].value_counts()
        num_unique_values = len(value_counts[value_counts == 1])

        # Fraction of explicitly missing cell values
        num_missing_values = df[column].isnull().sum()

        # Fraction of alphabetical cell values
        pattern = re.compile(r'[A-Za-z]+')
        num_alphabetical_values = sum(
            df.iloc[:, index].astype(str)
                             .apply(lambda x: bool(pattern.match(x)))
                             .astype(int)
        )

        # Fraction of numerical cell values
        pattern = re.compile(r'(([0-9]+)|(([0-9]+)\.([0-9]+)))')
        num_numerical_values = sum(
            df.iloc[:, index].astype(str)
                             .apply(lambda x: bool(pattern.match(x)))
                             .astype(int)
        )

        # Fraction of punctuation cell values
        num_punctuation_values = sum(
            df.iloc[:, index].astype(str)
                             .apply(lambda x: bool(x in string.punctuation))
                             .astype(int)
        )

        # Fraction of miscellaneous cell values
        num_miscellaneous_values = df[column].nunique()

        column_profile = np.asarray([
            num_unique_values, num_missing_values, num_alphabetical_values,
            num_numerical_values, num_punctuation_values, num_miscellaneous_values,
        ])/num_observations

        features = np.vstack([features, column_profile])

    # Remove first row of zeros before returning the feature dataframe
    return pd.DataFrame(data=features[1:].T, index=range(1, 7), columns=df.columns)

def create_column_profiles(datasets, profile_type, add_dataset_name=True):
    """Creates a profile for the the columns of the datasets.

    The parameter profile_type specifies which features should be used: DISTRIBUTION uses the
    distribution of characters in the columns, STRUCTURE_FEATURES uses structure features.

    Parameters:
        datasets (list[saged.datasets.Dataset]): List of datasets
        profile_type (saged.configuration.ProfileType): Which features to use
        add_dataset_name (bool, optional): If True, rename the column names in the resulting
            dataframe to tuples (dataset-name, column-name), defaults to True

    Returns:
        pandas.core.frame.DataFrame: A pandas DataFrame containing the distribution.
    """
    # Set profiler
    if profile_type is ProfileType.DISTRIBUTION:
        profiler = create_character_distributions
    elif profile_type is ProfileType.STRUCTURE_FEATURES:
        profiler = create_structure_features
    else:
        raise TypeError(f"Unknown type of profile '{profile_type}'.")

    # Get profiles for each dataset and (optionally) add the dataset name as multiindex level
    if add_dataset_name:
        profiles = [
            pd.concat({dataset.name: profiler(dataset.dirty_df)}, axis="columns")
            for dataset in datasets
        ]
    else:
        profiles = [profiler(dataset.dirty_df) for dataset in datasets]

    # Concatenate to one profile
    return pd.concat(profiles, axis="columns").fillna(0)

def get_profiles(dirty_dataset, historical_datasets, config, verbose=False):
    """Create column profiles for the dirty and historical datasets.

    Parameters:
        dirty_dataset (saged.datasets.Dataset): Dirty dataset
        historical_datasets (list[saged.datasets.Dataset]): Historical datasets.
        config (saged.configuration.Configuration): Configuration.
        verbose (bool, optional): If True, print verbose messages. Defaults to False.

    Returns:
        tuple: a tuple with the column profiles for the dirty and historical datasets
    """
    if verbose:
        print("* Create column profiles... ", end="", flush=True)

    dirty = create_column_profiles([dirty_dataset], config.profile_type, add_dataset_name=False)
    historical = create_column_profiles(historical_datasets, config.profile_type)

    if verbose:
        print("done.")

    return dirty, historical
