###################################################
# HoloClean: implement the HoloClean error detector

# Authors: Mohamed Abdelaal, Christian Hammacher
# Date: July 2022
# Software AG
# All Rights Reserved
###################################################

import time
import os
import pandas as pd

from baseline.setup.utils import store_detections
from baseline.setup.detectors.holoclean import HoloClean
from baseline.setup.detectors.holoclean.detect import NullDetector, ViolationDetector


def holoclean(dirty_df, dataset_name, dataset_path, detections_path):
    """
    Finds attributes in the dataframe that violate the defined denial constraints.

    @ Setup:

    ##### a. Installing PostgreSQL

    On Ubuntu, install PostgreSQL by running
    $ sudo apt update
    $ apt-get install postgresql postgresql-contrib

    ##### b. Setting up PostgreSQL for HoloClean

    By default, HoloClean needs a database `holo` and a user `holocleanuser` with permissions on it.

    1. Activate the PostgreSQL server
    $ sudo service postgresql start

    2. Create a database `holo` and user `holocleanuser`
       After navigating to the holoclean directory, run the following scripts
    $ bash create_db_ubuntu.sh
    $ bash create_pd_user_ubuntu.sh

    You can connect to the `holo` database from the PostgreSQL `psql` console by running
    `psql -U holocleanuser -W holo`.

    HoloClean currently populates the database `holo` with auxiliary and meta tables.
    To clear the database simply connect as a `root` user or as `holocleanuser` and run
    ```sql
    DROP DATABASE holo;
    CREATE DATABASE holo;
    ```

    @arguments:
    dirtyDF -- dataframe of shape n_R (# of records) x n_A (# of attributes) - containing a dirty version of a dataset
    dataset (String) -- name of the dataset

    @returns:
    detection_dictionary -- dictionary - keys represent i,j of dirty cells & values are constant string "JUST A DUUMY VALUE"
    """

    start_time = time.time()

    # 1. Setup a HoloClean session.
    hc = HoloClean(
        db_name='holo',
        domain_thresh_1=0,
        domain_thresh_2=0,
        weak_label_thresh=0.99,
        max_domain=10000,
        cor_strength=0.6,
        nb_cor_strength=0.8,
        epochs=10,
        weight_decay=0.01,
        learning_rate=0.001,
        threads=1,
        batch_size=1,
        verbose=False,
        timeout=3 * 60000,
        feature_norm=False,
        weight_norm=False,
        print_fw=True
    ).session

    # Define the path to the constaints directory
    dir = os.path.join(dataset_path, "constraints")

    try:
        all_constraints_file = open(os.path.join(dir, "_all_constraints.txt"), 'w+')
        all_constraints_file.truncate()
        for filename in os.listdir(dir):
            if filename.endswith(".txt") and not filename.startswith("_"):
                with open(os.path.join(dir, filename), "r") as infile:
                    for line in infile:
                        all_constraints_file.write(line)
        all_constraints_file.close()
    except:
        print("No constraints exist for the {} dataset".format(dataset_name))

    # 2. Load training data and denial constraints. Pass copy of dirtydf as load_data alters the parameter df
    copy_dirtydf = dirty_df.copy()
    hc.load_data(dataset_name, '', df=copy_dirtydf)
    hc.load_dcs(os.path.join(dir, "_all_constraints.txt"))
    hc.ds.set_constraints(hc.get_dcs())

    # detect errors with violation detector
    detectors = [NullDetector(), ViolationDetector()]
    errors_df = hc.detect_errors(detectors, return_errors=True)

    # transform detected errors from dataframe to detection dictionary
    detection_dictionary = {}
    for index, row in errors_df.iterrows():
        detection_dictionary[(row['_tid_'], dirty_df.columns.get_loc(row['attribute']))] = "JUST A DUMMY VALUE"

    # get runtime
    error_detect_runtime = time.time() - start_time

    # store detections in detector directory
    store_detections(detection_dictionary, detections_path)

    return detection_dictionary, error_detect_runtime
