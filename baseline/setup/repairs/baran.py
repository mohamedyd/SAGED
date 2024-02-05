###################################################################
# Baran: implement BARAN to repair errors in a dataset

# Authors: Mohamed Abdelaal, Christian Hammacher
# Date: July 2022
# Software AG
# All Rights Reserved
###################################################################

import os
import tempfile

from baseline.setup.detectors.raha.raha.correction import Correction as Raha_Correction
from baseline.setup.detectors.raha.raha.dataset import Dataset as Raha_Dataset


def baran(dirtyDF, cleanDF, detections, target_path):
    """
    runs baran detection from raha project. Repairs cells through sampling tuples/rows that are then
    labeled by a human (simulated through groundtruth). Labeling budget determines number of tuples labeled through gt.
    """

    # store dirty and clean in tmp file such that baran can read it again
    dirty_path = os.path.join(tempfile.gettempdir(), "dirty" + ".csv")
    dirtyDF.to_csv(dirty_path, sep=",", header=True, index=False)

    clean_path = os.path.join(tempfile.gettempdir(), "clean" + ".csv")
    cleanDF.to_csv(clean_path, sep=",", header=True, index=False)

    # dict to process raha steps internally
    internal_dataset_dict = {
        "name": "",
        "path": dirty_path,
        "clean_path": clean_path,
    }

    app = Raha_Correction()
    app.LABELING_BUDGET = int(0.01*dirtyDF.shape[0])  # rows that will be labeled from groundtruth (1% of all rows)
    app.VERBOSE = False
    app.SAVE_RESULTS = False

    # simulate detector initialization
    d = Raha_Dataset(internal_dataset_dict)
    d.dictionary = internal_dataset_dict

    # set detections for dataset instance
    d.detected_cells = detections
    d.has_ground_truth = True

    # initialize dataset for correcting with simulated dataset instance of detecting
    d = app.initialize_dataset(d)

    # initialize the Error Corrector Models
    app.initialize_models(d)

    # label tuples with ground truth, update models, generate features, predict correction
    while len(d.labeled_tuples) < app.LABELING_BUDGET:
        app.sample_tuple(d)
        if d.has_ground_truth:
            app.label_with_ground_truth(d)

        app.update_models(d)
        app.generate_features(d)
        app.predict_corrections(d)

    os.remove(dirty_path)
    os.remove(clean_path)

    d.create_repaired_dataset(d.corrected_cells)

    # Store the repaired dataset
    d.repaired_dataframe.to_csv(target_path, index=False, encoding="utf-8")

    return d.repaired_dataframe


