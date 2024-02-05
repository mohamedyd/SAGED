####################################################################
# Test the impact of the historical data on the performance of SAGED
# Authors: Mohamed Abdelaal
# Date: March 2023
# Software AG
# All Rights Reserved
####################################################################

import csv
import os
import time
import mlflow
import statistics
import argparse
from numpy import mean
import pandas as pd
from saged.saged import saged
from saged.datasets import Dataset
from itertools import combinations, chain

parser = argparse.ArgumentParser(description="SAGED: Software AG Error Detection")

parser.add_argument("--tags", type=str, default="None")
parser.add_argument("--dirty-dataset", required=True, help="dataset to detect errors in")
parser.add_argument("--historical-datasets", nargs="+", help="historical datasets providing design time knowledge")
parser.add_argument("--num-labels", type=int, default=20, help="number of labels to use")
parser.add_argument("--n-clusters", type=int, default=1, help="number of clusters to create when picking historical classifiers (mutually exclusive with --n-meta-features)")
parser.add_argument("--n-meta-features", type=int, default=0, help="number of meta features to create for each column (mutually exclusive with --n-clusters)")
parser.add_argument("-r", "--runs", type=int, default=1, help="number of times the method should run (results will be pretty-printed in a table)")
parser.add_argument("-v", "--verbose", action="store_true", default=False, help="print verbose messages")

args = parser.parse_args()

# Start an MLflow experiment
mlflow.set_experiment("Historical Datasets")

# Load all historical datasets
if args.historical_datasets is None:
    historical_datasets = [dataset.name for dataset in Dataset.load_all(skip=[args.dirty_dataset])]
else:
    historical_datasets = args.historical_datasets

# Generate combinations of historical data
combs = chain.from_iterable(combinations(historical_datasets, r) for r in range(len(historical_datasets) + 1))
historical_combinations = [list(comb) for comb in combs]

for hist_comb in historical_combinations:
    if hist_comb:
        print(hist_comb)
        list_precision, list_recall, list_f1_score, list_total_time = saged(dirty_dataset=args.dirty_dataset,
                                                                            historical_datasets=hist_comb,
                                                                            n_clusters=args.n_clusters,
                                                                            n_meta_features=args.n_meta_features,
                                                                            runs=args.runs,
                                                                            num_labels=args.num_labels,
                                                                            verbose=args.verbose
                                                                        ) 
        # Log relevant information about the experiment
        with mlflow.start_run():
            # log model parameters
            mlflow.log_param("Dirty Dataset", args.dirty_dataset)
            mlflow.log_param("Number of Runs", args.runs)
            mlflow.log_param("Historical Datasets", hist_comb)

            # log model metrics
            mlflow.log_metric("Average Precision", mean(list_precision))
            mlflow.log_metric("Average Recall", mean(list_recall))
            mlflow.log_metric("Average F1 Score", mean(list_f1_score))
            mlflow.log_metric("Average Total Time", mean(list_total_time))
            if args.runs > 1:
                mlflow.log_metric("Precision Std", statistics.stdev(list_precision))
                mlflow.log_metric("Recall Std", statistics.stdev(list_recall))
                mlflow.log_metric("F1 Score Std", statistics.stdev(list_f1_score))
                mlflow.log_metric("Total Time Std", statistics.stdev(list_total_time))