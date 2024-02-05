###################################################################
# Repair: implement a method which executes different repair methods

# Usage:
#  repair(clean_path, dirty_path, target_path, detections_path, dataset_name, repair_method):

# Authors: Mohamed Abdelaal, Christian Hammacher
# Date: July 2022
# Software AG
# All Rights Reserved
###################################################################

import os
from enum import Enum

import pandas as pd
from baseline.setup.repairs.mlImputer import mlImputer
from baseline.setup.repairs.standardImputer import standardImputer

from baseline.setup.repairs.baran import baran
from baseline.setup.repairs.cleanWithGroundTruth import cleanWithGroundTruth
from baseline.setup.repairs.activeClean import activeClean
from baseline.setup.utils import load_detections, get_all_errors, create_target_path, create_detections_path
from baseline.setup.detectors.detect_method import DetectMethod


class RepairMethod(Enum):
    ML_IMPUTER = 'mlImputer'
    STANDARD_IMPUTER = 'standardImputer'
    BARAN = "baran"
    CLEAN_GROUNDTRUTH = 'cleanWithGroundTruth'
    ACTIVECLEAN = 'activeClean'

    def __str__(self):
        return self.value


def repair(clean_path, dirty_path, target_path, detections_path, dataset_name, repair_method):
    """
    Repair a dataset via replacing detected dirty instances with generated values. Several repair methods can be
    invoked to generate the new values.
    @arguments:
    clean_path -- string, path to the clean dataset
    dirty_path -- string, path to the dirty dataset
    target_path -- string, path to the repaired data
    repair_method -- attribute of the RepairMethod class, define which repair method to use
    detection_path -- string, path to the file containing the indices of the detected dirty instances
    """

    # Initialize an empty dataframe for the repaired data
    df_repaired = pd.DataFrame()

    # Load the dirty data and its ground truth
    dirty_df = pd.read_csv(dirty_path, header="infer", encoding="utf-8", low_memory=False)
    clean_df = pd.read_csv(clean_path, header="infer", encoding="utf-8", low_memory=False)

    if os.path.exists(detections_path):
        detections_dictionary = load_detections(detections_path)
    else:
        raise ValueError("No detections found")

    if repair_method == RepairMethod.ML_IMPUTER:
        df_repaired = mlImputer(dirty_df, detections_dictionary, repair_method="separate", target_path=target_path,
                               num='decisionTree', cat='missForest')
        #df_repaired = mlImputer(dirty_df, detections_dictionary, repair_method="mix", target_path=target_path,
        #                       mix_method='missforest')
    elif repair_method == RepairMethod.STANDARD_IMPUTER:
        df_repaired = standardImputer(dirty_df, detections_dictionary, repair_method="impute", target_path=target_path,
                                      num='mean', cat='dummy')
    elif repair_method == RepairMethod.BARAN:
        df_repaired = baran(dirty_df, clean_df, detections_dictionary, target_path)
    elif repair_method == RepairMethod.CLEAN_GROUNDTRUTH:
        df_repaired = cleanWithGroundTruth(dirty_df, clean_df, target_path, detections_dictionary)
    elif repair_method == RepairMethod.ACTIVECLEAN:
        activeClean(dirty_df=dirty_df, clean_df=clean_df, dataset_name=dataset_name,
                    detections=detections_dictionary, sampling_budget=0.2)
    else:
        raise NotImplementedError

    return df_repaired

if __name__ == "__main__":
    # Get the data path
    dataset_name = 'breast_cancer'
    detector_name = DetectMethod.SAGED.__str__()
    repair_method = RepairMethod.ML_IMPUTER

    EXP_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..", "experiments"))
    DATA_PATH = os.path.abspath(os.path.join(EXP_PATH, "data"))

    # Retrieve the dirty and clean data
    clean_path = os.path.abspath(os.path.join(DATA_PATH, dataset_name, 'clean.csv'))
    dirty_path = os.path.abspath(os.path.join(DATA_PATH, dataset_name, 'dirty.csv'))
    # Prepare paths to store the results and intermediate data

    detections_path = create_detections_path(EXP_PATH, dataset_name, detector_name)
    target_path = create_target_path(EXP_PATH, dataset_name, detector_name, repair_method.__str__())

    repair(clean_path, dirty_path, target_path, detections_path, dataset_name, repair_method)
