#########################################################################
# Train ML models on data repaired while using SAGED for detecting errors
# Authors: Mohamed Abdelaal
# Date: March 2023
# Software AG
# All Rights Reserved
#########################################################################

import os
import argparse
from baseline.setup.detectors.detect_method import DetectMethod, DATA_PATH, EXP_PATH
from baseline.setup.repairs.repair import RepairMethod
from baseline.setup.utils import create_target_path, create_detections_path
from saged.train_e2e_saged import train_e2e_saged


# Initialize an argument parser
parser = argparse.ArgumentParser()
# Add the parser's options
parser.add_argument('--dirty-dataset',  nargs='+', default=None, required=True)
parser.add_argument('--repair-method',  nargs='+', type=RepairMethod, choices=list(RepairMethod), default=None)
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
parser.add_argument("-a", "--label-augmentation", choices=["random", "prediction", "active_learning", "knn_shapley"], default=None, help="use initial predictions to augment training data of the meta classifiers")
parser.add_argument("-o", "--output-file", help="csv output file to store results in")
parser.add_argument("-v", "--verbose", action="store_true", default=False, help="print verbose messages")
parser.add_argument('--tune-hyperparams', action='store_true')
parser.add_argument('--epochs', type=int, default=500)
args = parser.parse_args()

# Use all available base detectors if no specific detectors are selected
involved_detector = DetectMethod.SAGED
involved_repairs = list(RepairMethod) if not args.repair_method else args.repair_method

if args.verbose:
    print("[INFO] The involved detectors are: {}".format(involved_detector))
    print("[INFO] The involved repair methods are: {}".format(involved_repairs))


for dataset_name in args.dirty_dataset:

    for repair_method in involved_repairs:

        # Prepare the paths
        dataset_path = os.path.abspath(os.path.join(DATA_PATH, dataset_name))
        clean_path = os.path.join(dataset_path, 'clean.csv')
        dirty_path = os.path.join(dataset_path, 'dirty.csv')
        detections_path = create_detections_path(EXP_PATH, dataset_name, involved_detector.__str__())
        det_results_path = create_detections_path(EXP_PATH, dataset_name, involved_detector.__str__(), store_detection_metrics=True)
        target_path = create_target_path(EXP_PATH, dataset_name, involved_detector.__str__(), repair_method.__str__())

        for _ in range(args.runs):

            train_e2e_saged(clean_path=clean_path,
                            dirty_path=dirty_path,
                            detections_path=detections_path,
                            target_path=target_path,
                            repair_method=repair_method,
                            dirty_dataset=dataset_name,
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
                            label_augmentation=args.label_augmentation,
                            runs=args.runs,
                            num_labels=args.num_labels,
                            verbose=args.verbose,
                            tune_params=args.tune_hyperparams,
                            epochs=args.epochs)

