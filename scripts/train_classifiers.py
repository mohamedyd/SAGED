import argparse
from saged.configuration import ClassifierType
from saged.datasets import Dataset
from saged.classification import train_classifiers


parser = argparse.ArgumentParser(description="Train error classifiers for the given datasets.")
parser.add_argument("--datasets", nargs="+")
parser.add_argument("--classifiers", nargs="+", required=True)
parser.add_argument("-v", "--verbose", action="store_true", default=False)
args = parser.parse_args()

# If --datasets is not given, import available datasets
if args.datasets is None:
    datasets = Dataset.load_all()
else:
    datasets = [Dataset(name) for name in args.datasets]

classifier_types = [ClassifierType(classifier_type) for classifier_type in args.classifiers]

for dataset in datasets:
    print(f"Dataset '{dataset.name}':")
    train_classifiers(dataset, classifier_types, verbose=True)
    print("---\n")
