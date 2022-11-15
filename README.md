SAGED: Meta Learning-based Error Detection Method for Structured Data
=====================================================================

In this repository, we introduce a novel method for automated error detection in structured data, referred to as SAGED (Software AG Error Detection). The core idea behind SAGED is to formulate the task of error detection as a classification problem. In this realm, we exploit the design-time artifacts while generating features required to train a detection classifier. To this end, meta-learning is utilized to transfer knowledge from a set of historical previously-repaired datasets to the new dirty datasets, i.e., the datasets to be cleaned. Specifically, SAGED consists of two phases, namely the knowledge extraction phase and the detection phase. In the former phase, we train a set of ML models to differentiate between erroneous and clean samples in the historical datasets. The latter phase begins with matching the new dirty dataset with a set of the historical datasets, before using the corresponding models to generate the feature vector for the meta-classifier. In this case, the features represent the predictions obtained from the base classifiers. To realize the adoption of meta learning in the proposed invention, two challenges, including the varying-length feature vector, and the irrelevant knowledge problem, have to be overcome. To this end, the invention implements a zero-padding mechanism and a clustering approach to group the base classifiers in the knowledge extraction phase. 

# Setup
Clone with submodules

```shell script
git clone http://audio.digitalbusinessplatform.de/gitlab/kompaki/cleanlearning/saged.git --recurse-submodules
```

Create a virtual environment and install requirements

```shell script
python3 -m venv venv 
source venv/bin/activate
pip3 install --upgrade setuptools
pip3 install --upgrade pip
pip3 install -e .
```

## Folder overview
This section introduces the content of each script in the project's directory.
- `saged/`: Files implementing the error detection method
- `datasets/`: Dataset with clean and dirty data
- `error_generator/`: Module to generate errors
- `scripts/`: Scripts to run the method and train classifiers  
## Usage

To run SAGED, you need to have enough classifiers trained. To train MLP classifiers for every dataset in the `datasets/` directory, run
```bash
python saged/train.py --classifiers mlp_classifier
```
You can also specify datasets to train classifiers for with the `--datasets` argument and choose different classifiers.

## Follow-up Ideas

* Enhancements
    - Improving the clustering performance via avoiding randomness (see https://scikit-learn.org/stable/modules/clustering.html for a list of cluster algorithms)
    - Extending the design-time knowledge via adding more datasets to the feature store
    - Improving detection accuracy via adding metadata to the meta-learner feature vector
    - Comparative study with StoA in terms of ML performance
    - Train meta-classifier on a validation set  no labels are needed when detecting errors

* Extensions
    - Automatically configure the hyperparameters, e.g., # clusters k, according to the metadata of new datasets
    - Recognizing the error type, e.g., outlier, rule violation, missing value, etc.
    - Data augmentation to improve base classifiers in case of datasets with low error rates
    - Pre-train the meta classifier to avoid active learning and acquiring labels
