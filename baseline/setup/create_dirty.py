###########################################################################
# Create_dirty: implement a method to inject different errors into datasets

# Usage:
# inject_errors(clean_df, error_types, params, muted_columns,data_path)

# Authors: Mohamed Abdelaal
# Date: June 2022
# Software AG
# All Rights Reserved
###########################################################################

import os
from copy import deepcopy
from enum import Enum

import numpy as np
import pandas as pd

# Import necessary libraries and packages
from error_generator import List_selected, Error_Generator, Explicit_Missing_Value, \
    Implicit_Missing_Value, Gaussian_Noise, Random_Active_Domain, Typo_Keyboard


class ErrorType(Enum):
    """Error types injected using the error generator module."""
    SWAPPING = "cell_swapping"
    IMPLICIT_MV = "implicit_mv"
    EXPLICIT_MV = "explicit_mv"
    NOISE = "gaussian_noise"
    TYPOS = "typo_keyboard"
    OUTLIERS = "outliers"

    @property
    def func(self):
        func_dict = {
            ErrorType.SWAPPING: Random_Active_Domain(),
            ErrorType.IMPLICIT_MV: Implicit_Missing_Value(),
            ErrorType.EXPLICIT_MV: Explicit_Missing_Value(),
            ErrorType.NOISE: Gaussian_Noise(),
            ErrorType.TYPOS: Typo_Keyboard()
        }
        return func_dict[self]

    def __str__(self):
        return self.value


def inject_outlier(df, label, outlier_rate=0.1, multiplier=3):
    """injects errors in outlier_rate cases only in num columns
    injected value ist previous value + -/+1 * log(1+rand(0,1)*3) therefore
    the minimum outlier is always ol_thresholds stds away from current value but maybe further.
    """

    # Extract the name of the numerical columns
    num_cols = df.select_dtypes("number").columns
    # Exclcude the labels
    num_cols = [col for col in num_cols if col != label]

    # Create a mask of zeros
    mask = np.zeros(df[num_cols].size)
    mask[:int(outlier_rate * df[num_cols].size)] = 1
    np.random.shuffle(mask)
    mask = mask.reshape((df[num_cols].shape[0], df[num_cols].shape[1]))

    for i, col in enumerate(num_cols):
        std = np.std(df[col])
        df[col] = df[col] + mask.T[i] * np.random.normal(loc=0.0, scale=std * multiplier, size=df.shape[0])

    return df


def generate_errors(dataset, error_types, percent, muted_columns):
    """
    This method is used to inject various type of errors, including
    - typos based on keyboards
        + Duplicate the character
        + Delete the character
        + Shift the character one keyboard space

    - typos base on butter-fingers
        + A python library to generate highly realistic typos (fuzz-testing)

    - explicit missing value
        + randomly one value will be removed

    - implicit missing value
        + one of median or mode of the active domain, randomly pick and
            replace with the selected value

    - Random Active domain
        + randomly one value from active domain will be replaced with the selected value

    - Similar based Active domain
        + the most similar value from the active domain is picked and replaced with the selected value

    - White noise (min=0, var=1)
        + white noise added to the selected value(for string value the noise add to asci code of them)

    - gaussian noise:
        + the Gaussian noise added to the selected value (for string value the noise add to asci code of them). in this method, you can specify the noise rate as well.
    """

    dataset_dataframe = dataset
    dataset_dataframe = dataset_dataframe.apply(lambda x: x.str.strip())
    dataset = [list(dataset_dataframe.columns.to_numpy())] + list(dataset_dataframe.to_numpy())
    # Obtain a copy of the original dataset
    dirty_dataset = deepcopy(dataset)

    # Choose a selector, either list_selected or value_selected
    selector = List_selected()
    # Initialize an error generator
    error_generator = Error_Generator()

    # Loop over all error types
    for error_type in error_types:
        # Choose a strategy
        strategy = error_type

        # Inject errors
        # Percentage : The amount of errors to be injected
        # Mute column: Columns that should be safe, i.e. away from the error generator proccess
        dirty_dataset = error_generator.error_generator(method_gen=strategy, selector=selector,
                                                        percentage=percent * 50,
                                                        dataset=dirty_dataset, mute_column=muted_columns)

    return dirty_dataset


def inject_errors(clean_df, error_types, params, muted_columns, data_path):
    """
    This method injects different types of errors in a dataset

    @arguments:
         clean_df -- dataframe of the dataset to be injected
         error_types -- list comprising the error types to be injected. To inject outliers, error_types = [
         ErrorType.OUTLIERS]. To inject other error types, error_types = [[ErrorType.EXPLICIT_MV.func]]
         params -- list encapsulating the error rate and the outlier degree
         muted_columns -- list of strings denoting the columns which should remain clean (not injected)
         data_path -- string denoting the path of the dataset

    """

    # Get a copy of the clean data
    df = clean_df.copy()

    # Extract the parameters of the error injection methods
    all_errors_percent, outlier_multiplier = params

    # Drop muted columns, which must not change, e.g., labels
    if muted_columns:
        df.drop(muted_columns, axis=1, inplace=True)

    # Estimate rate of each error type
    number_error_types = len(error_types)
    error_rate = all_errors_percent / number_error_types

    # Initialize the dirty dataframe
    dirty_df = df.copy()

    for error_type in error_types:
        if error_type == ErrorType.OUTLIERS:
            dirty_df = inject_outlier(dirty_df, muted_columns, error_rate, outlier_multiplier)
        else:
            dirty_df = generate_errors(dirty_df.astype(str), error_type, error_rate, muted_columns)
            dirty_df = pd.DataFrame(dirty_df[1:], columns=dirty_df[0])

    # Restore the muted columns
    if muted_columns:
        muted_data = clean_df[muted_columns]
        dirty_df = dirty_df.join(muted_data)

    output_path = os.path.abspath(os.path.join(data_path, "dirty.csv"))
    dirty_df.to_csv(output_path, sep=",", index=False, encoding="utf-8")


if __name__ == "__main__":
    # Test the implementation
    # Get the data path
    DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "experiments", "data", "nasa"))
    df_clean = pd.read_csv(os.path.abspath(os.path.join(DATASET_PATH, "clean.csv")), index_col=False)
    method = [ErrorType.OUTLIERS, [ErrorType.EXPLICIT_MV.func]]
    dirty_df = inject_errors(clean_df=df_clean, error_types=method, params=[0.5, 1], muted_columns=['sound_pressure_level'],
                             data_path=DATASET_PATH)