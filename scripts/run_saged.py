import csv
import os
import time
import statistics
import argparse
from numpy import mean
from prettytable import PrettyTable
from saged.configuration import (
    ClassifierType, ClusterAlgorithm, Configuration, LabelingMethod, ProfileType
)
from saged.datasets import Dataset
from saged.meta import create_meta_features, evaluate, train_meta_classifiers
from saged.labeling import active_learning, clustering_based_sampling, random_split

parser = argparse.ArgumentParser(description="SAGED: Software AG Error Detection")
parser.add_argument("--dirty-dataset", required=True)
parser.add_argument("--historical-datasets", nargs="+")
parser.add_argument("--profile", default="structure_features")
parser.add_argument("--classifier", default="mlp_classifier")
parser.add_argument("--clustering", default="kmeans")
parser.add_argument("--labeling", default="none")
parser.add_argument("--propagate-labels", action="store_true", default=False)
parser.add_argument("--num-labels", type=int, default=20)
parser.add_argument("--n-clusters", type=int, default=1)
parser.add_argument("-r", "--runs", type=int, default=1)
parser.add_argument("-o", "--output-file")
parser.add_argument("-v", "--verbose", action="store_true", default=False)
args = parser.parse_args()

# Load dirty and historical datasets
dirty_dataset = Dataset(args.dirty_dataset)

# If --historical-datasets is not given, import available datasets
if args.historical_datasets is None:
    historical_datasets = Dataset.load_all(skip=[args.dirty_dataset])
else:
    historical_datasets = [Dataset(hd) for hd in args.historical_datasets]

# Get configuration
config = Configuration(
    profile_type=ProfileType(args.profile.lower()),
    classifier_type=ClassifierType(args.classifier.lower()),
    cluster_algorithm=ClusterAlgorithm(args.clustering.lower()),
    labeling_method=LabelingMethod(args.labeling.lower()),
    n_clusters=args.n_clusters
)

# Statistics
list_f1_score = []
list_recall = []
list_precision = []
list_total_time = []

for i in range(args.runs):
    if args.verbose:
        print(f"Iteration {i+1}")
        print("="*len(f"Iteration {i+1}"))

    start_total = time.time()

    # Create meta features
    if args.verbose:
        print("Create meta features:")

    meta_features = create_meta_features(
        dirty_dataset, historical_datasets, config, verbose=args.verbose
    )

    if args.verbose:
        print("Meta features created.\n")

    if args.verbose:
        print(f"Label samples (method: {config.labeling_method.value}):")

    if config.labeling_method is LabelingMethod.NONE:
        X_train, X_test, y_train, y_test = random_split(
            meta_features, dirty_dataset.get_actual_errors(), num_labels=args.num_labels
        )
    elif config.labeling_method is LabelingMethod.CLUSTERING:
        X_train, X_test, y_train, y_test = clustering_based_sampling(
            meta_features, dirty_dataset.get_actual_errors(), propagate=args.propagate_labels,
            num_labels=args.num_labels, verbose=args.verbose
        )
    elif config.labeling_method is LabelingMethod.ACTIVE_LEARNING:
        X_train, X_test, y_train, y_test = active_learning(
            meta_features, dirty_dataset.get_actual_errors(), config.classifier_type,
            num_labels=args.num_labels, verbose=args.verbose
        )

    if args.verbose:
        print("Samples labeled.\n")

    if args.verbose:
        print("Train meta classifiers:")

    meta_classifiers = train_meta_classifiers(
        X_train, y_train, config.classifier_type, verbose=args.verbose
    )

    if args.verbose:
        print("Meta classifiers trained.\n")

    if args.verbose:
        print("Evaluate classifiers:")

    precision, recall, f1_score = evaluate(meta_classifiers, X_test, y_test)

    end_total = time.time()
    time_total = end_total - start_total

    if args.verbose and args.runs > 1:
        print(f"* Precision: {precision}, recall: {recall}, F1 score: {f1_score}, ",
              f"total time: {time_total}\n")

    list_precision.append(precision)
    list_recall.append(recall)
    list_f1_score.append(f1_score)
    list_total_time.append(time_total)

print(f"\nDirty dataset: {dirty_dataset}")
print("Historical datasets:\n * {}".format('\n * '.join([str(hd) for hd in historical_datasets])))
print("Total amount of elements in historical datasets: ", end="")
print(sum(hd.dirty_df.size for hd in historical_datasets))

table = PrettyTable([""] + list(range(1, args.runs+1)) + ["Mean values"])
table.add_row(["Precision"] + list_precision + [mean(list_precision)])
table.add_row(["Recall"] + list_recall + [mean(list_recall)])
table.add_row(["F1 score"] + list_f1_score + [mean(list_f1_score)])
table.add_row(["Total time"] + list_total_time + [mean(list_total_time)])
table.align = "r"
table.float_format = ".2"
print(table)

if args.runs > 1:
    print("Precision standard deviation: ", statistics.stdev(list_precision))
    print("Recall standard deviation: ", statistics.stdev(list_recall))
    print("F1-Score standard deviation: ", statistics.stdev(list_f1_score))
    print("Total Time standard deviation: ",  statistics.stdev(list_total_time))

if args.output_file is not None:
    if not os.path.exists(os.path.dirname(args.output_file)):
        os.makedirs(os.path.dirname(args.output_file))

    with open(args.output_file, 'w+', encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["precision", "recall", "f1 score", "total time"])
        writer.writerows(zip(list_precision, list_recall, list_f1_score, list_total_time))
