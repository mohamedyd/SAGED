########################################################################################################
# Evaluate: implement a set of methods to evaluate the performance of error detection and repair methods

# Authors: Mohamed Abdelaal, Christian Hammacher
# Date: July 2022
# Software AG
# All Rights Reserved
#########################################################################################################

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


def evaluate_detector(all_errors, detections):
    """
    Evaluate the performance (accuracy) of error detection methods in terms of precision, recall, and F1 scores

    @arguments:
    all_errors -- dictionary, all errors in a dataset found via comparing the dirty dataset and its ground truth
    detections -- dictionary, keys represent i,j of dirty cells & values are string "JUST A DUUMY VALUE"

    @return:
    precision, recall, f1 -- floats, accuracy metrics
    """
    tp = 0.0
    output_size = 0.0

    # for key, value in self.actual_errors.items():
    #    logging.info("actual_errors ", key, value)

    for cell in detections:
        output_size = output_size + 1
        if cell in all_errors:
            tp = tp + 1

    precision = 0.0 if output_size == 0 else tp / output_size
    recall = 0.0 if len(all_errors) == 0 else tp / (len(all_errors))
    f1 = 0.0 if (precision + recall) == 0 else (2 * precision * recall) / (precision + recall)

    return precision, recall, f1


def evaluate_repair(clean_df, dirty_df, repaired_df, detections_dictionary):
    """
    Run two experiments for evaluating repaired datasets, as follows:
    1. Only numerical: get numerical columns from groundtruthDF and calulate RMSE for dirty dataset and for the
    repaired dataset (considers cells where groundtruth, dirty and repaired dataset have numerical values).
    2. Only categorical: get categorical columns from groundtruthDF and calculate precision, recall, f1 relative to all
    errors in the respective columns

    Arguments:
    detections_dictionary -- dictionary, keys represent i,j of dirty cells & values are string "JUST A DUUMY VALUE"
    clean_df -- dataframe, ground truth version of the dataset
    dirtyDF -- dataframe, dirty version of the dataset
    cleanedDF -- dataframe, repaired version of the dataset
    """

    # Initialize a dictionary to pack the results
    evaluation_dict = {}

    # Get numerical and categorical columns of the ground truth
    groundTruthDF = clean_df.apply(pd.to_numeric, errors="ignore")
    gt_num_columns = groundTruthDF.select_dtypes(include="number").columns
    gt_cat_columns = groundTruthDF.select_dtypes(exclude="number").columns

    # ===================================================================================
    # Extract metadata of ground truth, repaired and dirty dataset
    # ===================================================================================

    # Return metrics not available if shapes do not equal, e.g., in case of deleting dirty tuples
    if dirty_df.shape != repaired_df.shape != clean_df.shape:

        evaluation_dict = {"gt_#cat_col": len(gt_cat_columns),
                           'gt_#num_col': len(gt_num_columns),
                           "repaired_#cat_col": len(repaired_df.select_dtypes(exclude="number").columns),
                           'repaired_#num_col': len(repaired_df.select_dtypes(include="number").columns),
                           "dirty_#cat_col": len(repaired_df.select_dtypes(exclude="number").columns),
                           'dirty_#num_col': len(repaired_df.select_dtypes(include="number").columns),
                           'onlyNum_rmse_repaired': None,
                           'onlyNum_rmse_dirty': None,
                           'onlyCat_total_repairs': None,
                           'onlyCat_tp': None,
                           'onlyCat_actual_#errors': None,
                           'onlyCat_p': None,
                           'onlyCat_r': None,
                           'onlyCat_f': None,
                           'models': None
                           }

    else:
        # =================================================================================
        # Experiment 1: Only Numerical
        # only numerical columns of ground truth, removes values from consideration for RMSE
        #     in dirty_df and repaired_df that are not numerical. Looks at overlap of cells where ground truth,
        #     repaired and dirty cells are numerical.
        # =================================================================================

        if len(gt_num_columns) != 0:
            y_groundtruth = groundTruthDF[gt_num_columns].to_numpy(dtype=float)
            y_cleaned = repaired_df[gt_num_columns].to_numpy()
            y_dirty = dirty_df[gt_num_columns].to_numpy()

            for (x, y), _ in np.ndenumerate(y_groundtruth):
                try:
                    y_cleaned[x, y] = float(y_cleaned[x, y])
                    y_dirty[x, y] = float(y_dirty[x, y])
                    y_groundtruth[x, y] = float(y_groundtruth[x, y])
                except:
                    y_cleaned[x, y] = np.nan
                    y_dirty[x, y] = np.nan
                    y_groundtruth[x, y] = np.nan

            scaler = StandardScaler()
            """ y_groundtruth, y_cleaned, y_dirty have nan at the same positions
                thus nan values can be simply removed and the resulting arrays still fit """

            # scale, remove nan values
            y_true = scaler.fit_transform(y_groundtruth).flatten().astype(float)
            y_true = y_true[np.logical_not(np.isnan(y_true))]

            # scale, remove nan values and calculate rmse for repaired dataset
            y_pred = scaler.fit_transform(y_cleaned).flatten().astype(float)
            y_pred = y_pred[np.logical_not(np.isnan(y_pred))]
            rmse_repaired = mean_squared_error(y_true, y_pred, squared=False)

            # scale, remove nan values and calculate rmse for dirty dataset
            y_pred = scaler.fit_transform(y_dirty).flatten().astype(float)
            y_pred = y_pred[np.logical_not(np.isnan(y_pred))]
            rmse_dirty = mean_squared_error(y_true, y_pred, squared=False)

        else:
            rmse_repaired, rmse_dirty = 0.0, 0.0

        # =============================================================================
        #  Experiment 2: Only Categorical
        #  calculate f1, precision, recall for only categorical columns of ground truth
        # =============================================================================

        tp_cat = 0.0
        cat_repairsCounter = 0
        actual_errors_cat_dict = {(row, col): value for (row, col), value in detections_dictionary.items() if
                                  clean_df.columns[col] in gt_cat_columns}
        for (row_i, col_i), dummy in detections_dictionary.items():

            # if detection is in a categorical column of groundtruth
            if repaired_df.columns[col_i] in gt_cat_columns:
                errors_in_cat = +1
                # check if repair has happend
                if repaired_df.iat[row_i, col_i] != dirty_df.iat[row_i, col_i]:
                    # counter all repairs
                    cat_repairsCounter = cat_repairsCounter + 1

                    # check if detected error was corretly repaired
                    if repaired_df.iat[row_i, col_i] == clean_df.iat[row_i, col_i]:
                        tp_cat = tp_cat + 1

        precision_cat = 0.0 if cat_repairsCounter == 0 else tp_cat / cat_repairsCounter
        recall_cat = 0.0 if len(actual_errors_cat_dict) == 0 else tp_cat / len(actual_errors_cat_dict)
        f1_cat = 0.0 if (precision_cat + recall_cat) == 0 else (2 * precision_cat * recall_cat) / (
                precision_cat + recall_cat)

        evaluation_dict = {
            "gt_#cat_col": len(gt_cat_columns),
            'gt_#num_col': len(gt_num_columns),
            "repaired_#cat_col": len(repaired_df.select_dtypes(exclude="number").columns),
            'repaired_#num_col': len(repaired_df.select_dtypes(include="number").columns),
            "dirty_#cat_col": len(dirty_df.select_dtypes(exclude="number").columns),
            'dirty_#num_col': len(dirty_df.select_dtypes(include="number").columns),

            'onlyNum_rmse_repaired': rmse_repaired,
            'onlyNum_rmse_dirty': rmse_dirty,

            'onlyCat_total_repairs': cat_repairsCounter,
            'onlyCat_tp': tp_cat,
            'onlyCat_actual_#errors': len(actual_errors_cat_dict),
            'onlyCat_p': precision_cat,
            'onlyCat_r': recall_cat,
            'onlyCat_f': f1_cat,

            'model': None
        }

    return evaluation_dict
