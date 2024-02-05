####################################################################################
# OutlierDetector: implement three outlier detection methods, namely IF, SD, and IQR

# Authors: Mohamed Abdelaal, Christian Hammacher
# Date: July 2022
# Software AG
# All Rights Reserved
###################################################################################


import time
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from baseline.setup.utils import store_detections
from baseline.setup.evaluate import evaluate_detector


def outlierdetector(dirtydf, detect_method, detections_path):
    """
    Detect outliers in numerical columns.

    Configuration options: set nstd for method SD, set k for method IQR, set contamination for Method IF

    Arguments:
    dirtyDF -- dataframe of shape n_R (# of records) x n_A (# of attributes) - containing a dirty version of a dataset
    detect_method (String) -- can be SD, IQR, IF

    Returns:
    detection_dictionary -- dictionary - keys represent i,j of dirty cells & values are constant string "JUST A DUUMY VALUE"

    """

    def SD(x, nstd=3.0):
        # Standard Deviaiton Method (Univariate)
        mean, std = np.mean(x), np.std(x)
        cut_off = std * nstd
        lower, upper = mean - cut_off, mean + cut_off
        return lambda y: (y > upper) | (y < lower)

    def IQR(x, k=1.5):
        # Interquartile Range (Univariate)
        q25, q75 = np.percentile(x, 25), np.percentile(x, 75)
        iqr = q75 - q25
        cut_off = iqr * k
        lower, upper = q25 - cut_off, q75 + cut_off
        return lambda y: (y > upper) | (y < lower)

    def IF(x, contamination=0.01):
        # Isolation Forest (Univariate)
        # IF = IsolationForest(contamination='auto')
        IF_model = IsolationForest(contamination=contamination)
        IF_model.fit(x.reshape(-1, 1))
        return lambda y: (IF_model.predict(y.reshape(-1, 1)) == -1)

    start_time = time.time()

    # Define the detection function
    detect_fn_dict = {'SD': SD, 'IQR': IQR, "IF": IF}
    detect_fn = detect_fn_dict[detect_method]

    # transform dirtdf which has dtype string to numeric type if possible
    dirtydf = dirtydf.apply(pd.to_numeric, errors="ignore")
    # Drop missing values, necessary for IF
    dirtydf.fillna(0, axis=0, inplace=True)

    num_df = dirtydf.select_dtypes(include='number')
    cat_df = dirtydf.select_dtypes(exclude='number')
    X = num_df.values
    m = X.shape[1]

    # calculate for each row the detector in form of a lambda expression
    detectors = []
    for i in range(m):
        x = X[:, i]
        detector = detect_fn(x)
        detectors.append(detector)

    ind_num = np.zeros_like(num_df).astype('bool')
    ind_cat = np.zeros_like(cat_df).astype('bool')

    # check for each column if respective lambda expression is true
    # if there is a outlier
    for i in range(m):
        x = X[:, i]
        detector = detectors[i]
        is_outlier = detector(x)
        ind_num[:, i] = is_outlier

    ind_num = pd.DataFrame(ind_num, columns=num_df.columns)
    ind_cat = pd.DataFrame(ind_cat, columns=cat_df.columns)
    ind = pd.concat([ind_num, ind_cat], axis=1).reindex(columns=dirtydf.columns)

    # create detection dict
    detection_dictionary = {}
    for col in ind.columns:

        col_j = ind.columns.get_loc(col)

        for i, row in ind.iterrows():

            # if cell is true it is outlier
            if ind.iat[i, col_j]:
                detection_dictionary[(i, col_j)] = "JUST A DUMMY VALUE"

    # get runtime
    error_detect_runtime = time.time() - start_time

    # store detections in detector directory
    store_detections(detection_dictionary, detections_path)

    return detection_dictionary, error_detect_runtime
