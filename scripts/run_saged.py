##########################################################
# Detect errors in dirty datasets using the SAGED detector
# Authors: Tim Ktitarev and Mohamed Abdelaal
# Date: March 2023
# Software AG
# All Rights Reserved
###########################################################


import csv
import os
import time
import mlflow
import statistics
import argparse
from numpy import mean
import pandas as pd
from saged.saged import saged



parser = argparse.ArgumentParser(description="SAGED: Software AG Error Detection")

parser.add_argument("--tags", type=str, default="None")
parser.add_argument("--dirty-dataset", required=True, help="dataset to detect errors in")
parser.add_argument("--historical-datasets", nargs="+", help="historical datasets providing design time knowledge")
parser.add_argument("--features", choices=["meta", "classic"], default="meta", help="which features to use (classic: see featurization.py)")
parser.add_argument("--profile", choices=["structure_features", "distribution"], default="structure_features", help="which profile to use (warning: currently clustering does not work with the distribution profile)")
parser.add_argument("--classifier", choices=["mlp_classifier"], default="mlp_classifier", help="type of classifier to choose from historical datasets (warning: currently only MLP classifier is implemented)")
parser.add_argument("--clustering", choices=["kmeans"], default="kmeans", help="type of clustering to use (warning: currently only KMeans is implemented)")
parser.add_argument("--labeling-strategy", choices=["none", "clustering", "active_learning", "heuristic"], default="none", help="which labeling strategy to use")
parser.add_argument("--similarity", choices=["clustering", "cosine"], default="clustering", help="which similarity measure to pick historical classifiers")
parser.add_argument("--propagate-labels", action="store_true", default=False, help="propagate labels (only makes sense if you're using clustering as similarity measure)")
parser.add_argument("--num-labels", type=int, default=20, help="number of labels to use")
parser.add_argument("--n-clusters", type=int, default=1, help="number of clusters to create when picking historical classifiers (mutually exclusive with --n-meta-features)")
parser.add_argument("--n-meta-features", type=int, default=0, help="number of meta features to create for each column (mutually exclusive with --n-clusters)")
parser.add_argument("-r", "--runs", type=int, default=1, help="number of times the method should run (results will be pretty-printed in a table)")
parser.add_argument("-o", "--output-file", help="csv output file to store results in")
parser.add_argument("-a", "--label-augmentation", choices=["random", "prediction", "active_learning", "knn_shapley"], default=None, help="use initial predictions to augment training data of the meta classifiers")
parser.add_argument("-v", "--verbose", action="store_true", default=False, help="print verbose messages")

args = parser.parse_args()

# Start an MLflow experiment
mlflow.set_experiment("SAGED")

list_precision, list_recall, list_f1_score, list_total_time = saged(dirty_dataset=args.dirty_dataset,
                                                                    historical_datasets=args.historical_datasets,
                                                                    features=args.features,
                                                                    profile=args.profile,
                                                                    classifier=args.classifier,
                                                                    clustering=args.clustering,
                                                                    propagate_labels=args.propagate_labels,
                                                                    labeling=args.labeling_strategy,
                                                                    similarity=args.similarity,
                                                                    n_clusters=args.n_clusters,
                                                                    n_meta_features=args.n_meta_features,
                                                                    runs=args.runs,
                                                                    num_labels=args.num_labels,
                                                                    label_augmentation=args.label_augmentation,
                                                                    verbose=args.verbose
                                                                    ) 

# Log relevant information about the experiment
with mlflow.start_run():
    # log model parameters
    mlflow.log_param("Dirty Dataset", args.dirty_dataset)
    mlflow.log_param("Number of Runs", args.runs)
    mlflow.log_param("Labeling Method", args.labeling_strategy)
    mlflow.log_param("ML Model", args.classifier)
    mlflow.log_param("Features", args.features)
    mlflow.set_tag("Explanation", args.tags)

    # Set the historical datasets to none if the classic features are used
    hist_datasets = [str(hd) for hd in args.historical_datasets] if args.features == "meta" else ""
    mlflow.log_param("Historical Datasets", hist_datasets)

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

if args.output_file is not None:
    if not os.path.exists(os.path.dirname(args.output_file)):
        os.makedirs(os.path.dirname(args.output_file))

    with open(args.output_file, 'w+', encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["precision", "recall", "f1 score", "total time"])
        writer.writerows(zip(list_precision, list_recall, list_f1_score, list_total_time))
