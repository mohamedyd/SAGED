###################################################################
# CleanWithGroundTruth: implement the cleanWithGroundTruth method

# Authors: Mohamed Abdelaal, Christian Hammacher
# Date: July 2022
# Software AG
# All Rights Reserved
###################################################################

def cleanWithGroundTruth(dirty_df, clean_df, target_path, detections_dictionary):
    """
    Replace detected dirty instances with values from the clean version of the dataset. This method is used to
    estimate the performance upper-bound.

    @arguments:
    detection_dictionary -- dictionary, keys represent i,j of dirty cells & values are string "JUST A DUUMY VALUE"
    dirtyDF -- dataframe, dirty version of the dataset
    clean_df -- dataframe, ground truth version of the dataset
    target_path -- string, path to the repaired dataset

    @return:
    repaired_df -- dataframe, repaired version of the dirty dataset
    """

    if len(detections_dictionary) == 0:
        raise ValueError

    # initialize cleaned data with dirty data
    repaired_df = dirty_df.copy()

    # iterate through detection_dictionary and set correct values at detected error cells
    for (row_i, col_i), dummy in detections_dictionary.items():
        # replace a dirty cell with its clean value
        repaired_df.iat[row_i, col_i] = clean_df.iat[row_i, col_i]

    # Store the repaired dataset
    repaired_df.to_csv(target_path, index=False, encoding="utf-8")

    return repaired_df
