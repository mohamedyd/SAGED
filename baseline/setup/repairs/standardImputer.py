###################################################################
# StandardImputer: implement a set of statistical imputation  methods

# Authors: Mohamed Abdelaal, Christian Hammacher
# Date: July 2022
# Software AG
# All Rights Reserved
###################################################################

import sys

import pandas as pd
import sklearn.neighbors._base

sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base


def standardImputer(dirty_df, detections, repair_method, target_path, **kwargs):
    """
    Repair a dirty dataset using either deletion or statistical-based imputation.

    @arguments:
    detection_dictionary -- dictionary, keys represent i,j of dirty cells & values are string "JUST A DUUMY VALUE"
    dirtyDF -- dataframe, dirty version of the dataset
    method -- string, "delete" per default but can be set to "impute"
    num -- string, needed when method="impute". It defines how to impute dirty instances in numerical attributes.
           Possible values: "mean", "median", "mode"
    cat -- string, needed when method="impute". it defines how to impute dirty instances in in categorical attributes
           Possible values: "mode", "dummy"

    @return:
    repairedDF -- dataframe,  repaired version of the dirty dataset
    """

    # If method is delete, drop all rows from detections keys (row, col)
    if repair_method == "delete":
        drop_rows = []
        for (row_i, col_i), dummy in detections.items():
            drop_rows.append(row_i)

        repaired_df = dirty_df.drop(drop_rows)

        # If method is impute, impute detected cells with respective strategy provided in **kwargs
    elif repair_method == "impute":

        num_method = kwargs["num"]
        cat_method = kwargs["cat"]

        # Transform dirty_df which has dtype string to numeric type if possible
        dirtydf = dirty_df.apply(pd.to_numeric, errors="ignore")

        num_df = dirtydf.select_dtypes(include="number")
        cat_df = dirtydf.select_dtypes(exclude="number")

        if num_method == "mean":
            num_imp = num_df.mean()
        elif num_method == "median":
            num_imp = num_df.median()
        elif num_method == "mode":
            num_imp = num_df.mode().iloc[0]
        else:
            raise NotImplemented

        if cat_method == "mode":
            cat_imp = cat_df.mode().iloc[0]
        elif cat_method == "dummy":
            cat_imp = ["missing"] * len(cat_df.columns)
            cat_imp = pd.Series(cat_imp, index=cat_df.columns)
        else:
            raise NotImplemented

        impute = pd.concat([num_imp, cat_imp], axis=0)

        repaired_df = dirty_df.copy()

        # for every entry in detections impute the value in repairedDF with
        # the impute value for the respective column
        for (row_i, col_i), dummy in detections.items():
            repaired_df.iat[row_i, col_i] = impute[repaired_df.columns[col_i]]
    else:
        raise NotImplemented

    # Store the repaired dataset
    repaired_df.to_csv(target_path, index=False, encoding="utf-8")

    return repaired_df
