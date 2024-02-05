###################################################################
# MlImputer: implement several ML-based imputation methods

# Authors: Mohamed Abdelaal, Christian Hammacher
# Date: July 2022
# Software AG
# All Rights Reserved
###################################################################

import logging
import sys

import numpy as np
import pandas as pd
#import sklearn.neighbors._base
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer as sklearnIterativeImputer, KNNImputer as sklearnKNNImputer
from sklearn.linear_model import BayesianRidge

from impyute.imputation.cs import em as impEM
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeRegressor

import sklearn.neighbors._base
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
#sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
from missingpy import MissForest
#from missforest.miss_forest import MissForest


def mlImputer(dirty_df, detections, repair_method, target_path, **kwargs):
    """
    Impute dirty instances using ml based imputation methods. It runs either separate imputation methods for
    categorical and numerical columns or one mixed imputation method for both. For mixed methods, a mix_method has to
    be provieded. For seperate methods, num and cat parameters have to be provided.
    To be installed: sklearn.impute, missingpy, datawig

    @arguments:
    detections -- dictionary, keys represent i,j of dirty cells & values are constant string "JUST A DUUMY VALUE"
    dirty_df -- dataframe, dirty version of a dataset
    repair_method -- string, "mix" oder "separate". "mix" means that a method is applied to all columns,
    "separate" means that separate methods get applied to categorical (cat) and numerical (num) columns
    mix_method -- string, method to be applied if repair_method == "mix". Available: "missForest", "datawig"
    num -- string, method to be applied for numerical columns if repair_method == "separate". Available:
    "missForest", "knn", "em", "decisionTree", "bayesianRidge", "extraTrees", "datawig"
    cat -- string, method to be applied for categorical columns if repair_method == "separate". Available:
    "missForest", "datawig"

    @return:
    repaired_df -- dataframe, repaired version of the dirty dataset
    """

    def encode_cat(X_c):
        data = X_c.copy()
        nonulls = data.dropna().values
        impute_reshape = nonulls.reshape(-1, 1)
        encoder = OrdinalEncoder()
        impute_ordinal = encoder.fit_transform(impute_reshape)
        data.loc[data.notnull()] = np.squeeze(impute_ordinal)
        return data, encoder

    def decode_cat(X_c, encoder):
        data = X_c.copy()
        nonulls = data.dropna().values.reshape(-1, 1)
        n_cat = len(encoder.categories_[0])
        nonulls = np.round(nonulls).clip(0, n_cat - 1)
        nonulls = encoder.inverse_transform(nonulls)
        data.loc[data.notnull()] = np.squeeze(nonulls)
        return data

    if repair_method == "mix":
        num_method = None
        cat_method = None
        mix_method = kwargs["mix_method"]
        save_extension = "mix-{}".format(mix_method)
    elif repair_method == "separate":
        mix_method = None
        num_method = kwargs["num"]
        cat_method = kwargs["cat"]
        save_extension = "separate-{}-{}".format(num_method, cat_method)
    else:
        logging.info("incorrect parameters for method")
        sys.exit(1)

    dirtyDF_nan = dirty_df.copy()

    # change all occurances detections to np.nan
    # imputers identify cells to impute by checking if they are np.nan
    for (row_i, col_i), dummy in detections.items():
        dirtyDF_nan.iat[row_i, col_i] = np.nan

    # transform dirtdf which has dtype string to numeric type if possible.
    # np.nan is float and thus is numeric
    dirtyDF_nan = dirtyDF_nan.apply(pd.to_numeric, errors="ignore")

    num_df_orig = dirtyDF_nan.select_dtypes(include="number")
    cat_df = dirtyDF_nan.select_dtypes(exclude="number")
    
    
    # for numerical columns save columns that are all nan and create
    # a new dataframe that excludes those columns
    num_all_nan_cols = []
    for col in num_df_orig.columns:
        if num_df_orig[col].isnull().sum() == num_df_orig.shape[0]:
            num_all_nan_cols.append(col)
    num_df = num_df_orig.drop(columns=num_all_nan_cols)

    if num_method == "knn":
        # assigns mean of n_neighbors closests values

        imputer = sklearnKNNImputer(missing_values=np.nan, n_neighbors=5)
        num_repaired = imputer.fit_transform(num_df)

        # repaired version of numerical columns that are not all nan
        num_repaired = pd.DataFrame(num_repaired, columns=num_df.columns)

    if num_method == "missForest":
        # runs missingpy's MissForest on numerical columns.
        # Impute value are based on other values in row (numerical columnns)

        #imputer = MissForest(missing_values=np.nan)
        imputer = MissForest()
        num_repaired = imputer.fit_transform(num_df)

        # repaired version of numerical columns that are not all nan
        num_repaired = pd.DataFrame(num_repaired, columns=num_df.columns)

    if num_method == "em":
        # runs expected maximization imputer from impyute library

        # impEM only works if there are any np.nan's in the dataset
        num_repaired = impEM(num_df.to_numpy().astype(np.float)) if num_df.isnull().values.any() else num_df
        # repaired version of numerical columns that are not all nan
        num_repaired = pd.DataFrame(num_repaired, columns=num_df.columns)

    if num_method == "decisionTree" or num_method == "bayesianRidge" or num_method == "extraTrees":

        # instantiate estimator
        if num_method == "decisionTree":
            estimator = DecisionTreeRegressor(max_features='sqrt')
        elif num_method == "bayesianRidge":
            estimator = BayesianRidge()
        elif num_method == "extraTrees":
            estimator = ExtraTreesRegressor(n_estimators=10)

        imputer = sklearnIterativeImputer(estimator=estimator, missing_values=np.nan)
        num_repaired = imputer.fit_transform(num_df)

        # repaired version of numerical columns that are not all nan
        num_repaired = pd.DataFrame(num_repaired, columns=num_df.columns)

    if cat_method == "missForest" and cat_df.shape[1] > 0:
        print(list(cat_df.columns))
        print(cat_df.index)
        print(cat_df.shape)
        # decodes categorical variables and runs missingpy's MissForest on
        # categorical columns. Impute value are based on other values in row (only categorical columns)
        
        # encode categorical columns
        cat_encoders = {}
        cat_X_enc = []
        for c in cat_df.columns:
            X_c_enc, encoder = encode_cat(cat_df[c])
            cat_X_enc.append(X_c_enc)
            cat_encoders[c] = encoder
        cat_X_enc = pd.concat(cat_X_enc, axis=1)
        cat_columns = cat_df.columns
        cat_indices = [i for i, c in enumerate(cat_X_enc.columns) if c in cat_columns]

        # impute np.nan values
        imputer = MissForest(missing_values=np.nan)
        #imputer = MissForest()

        #cat_repaired_enc = imputer.fit_transform(cat_X_enc.values.astype(float), cat_vars=cat_indices)
        cat_repaired_enc = imputer.fit_transform(cat_X_enc.values.astype(float))
        cat_repaired_enc = pd.DataFrame(cat_repaired_enc, columns=cat_X_enc.columns)

        # decode encoded representation
        cat_X_imp = cat_repaired_enc
        cat_X_dec = []
        for c in cat_df.columns:
            X_c_dec = decode_cat(cat_X_imp[c], cat_encoders[c])
            cat_X_dec.append(X_c_dec)
        cat_X_dec = pd.concat(cat_X_dec, axis=1)

        # repaired version of categorical columns that are not all nan
        cat_repaired = cat_X_dec


    if mix_method == "missForest":
        # decodes categorical variables and runs missingpy's MissForest on
        # all columns (numerical = cateogircal). Impute value are based on other values in row (numerical + cateogrical columnns)
        
        # only if there are any categorical columns
        if cat_df.shape[1] > 0:
            cat_encoders = {}
            cat_X_enc = []
            for c in cat_df.columns:
                X_c_enc, encoder = encode_cat(cat_df[c])
                cat_X_enc.append(X_c_enc)
                cat_encoders[c] = encoder
            cat_X_enc = pd.concat(cat_X_enc, axis=1)
            X_enc = pd.concat([num_df, cat_X_enc], axis=1)  # because mix_method
            cat_columns = cat_df.columns
            cat_indices = [i for i, c in enumerate(X_enc.columns) if c in cat_columns]
        else:
            X_enc = num_df  # because mix method
            cat_indices = None

        # impute np.nan values
        imputer = MissForest()
        repaired_enc = imputer.fit_transform(X_enc.values.astype(float))
        repaired_enc = pd.DataFrame(repaired_enc, columns=X_enc.columns)

        if cat_df.shape[1] > 0:
            # decode encoded representation
            num_X_imp = repaired_enc[num_df.columns]  # new
            cat_X_imp = repaired_enc[cat_df.columns]  # new
            cat_X_dec = []
            for c in cat_df.columns:
                X_c_dec = decode_cat(cat_X_imp[c], cat_encoders[c])
                cat_X_dec.append(X_c_dec)
            cat_X_dec = pd.concat(cat_X_dec, axis=1)
            X_dec = pd.concat([num_X_imp, cat_X_dec], axis=1)

        # repaired version of categorical columns that are not all nan
        #repaired = X_dec


    repaired_df = dirty_df.copy()
    if repair_method == "mix":
        outoforder_concat = pd.concat([X_dec, dirty_df[num_all_nan_cols]], axis=1)
    else:

        if not cat_df.shape[1] > 0:
            outoforder_concat = pd.concat([num_repaired, dirty_df[num_all_nan_cols]], axis=1)
        else:
            outoforder_concat = pd.concat([num_repaired, cat_repaired, dirty_df[num_all_nan_cols]], axis=1)
    for col in outoforder_concat.columns:
        repaired_df[col] = outoforder_concat[col]

    # Store the repaired dataset
    repaired_df.to_csv(target_path, index=False, encoding="utf-8")

    return repaired_df
