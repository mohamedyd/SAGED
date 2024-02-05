SAGED: Meta Learning-based Error Detection Method for Structured Data
=====================================================================

In this repository, we introduce a novel method for automated error detection in structured data, referred to as SAGED (Software AG Error Detection). The core idea behind SAGED is to formulate the task of error detection as a classification problem. In this realm, we exploit the design-time artifacts while generating features required to train a detection classifier. To this end, meta-learning is utilized to transfer knowledge from a set of historical previously-repaired datasets to the new dirty datasets, i.e., the datasets to be cleaned. Specifically, SAGED consists of two phases, namely the knowledge extraction phase and the detection phase. In the former phase, we train a set of ML models to differentiate between erroneous and clean samples in the historical datasets. The latter phase begins with matching the new dirty dataset with a set of the historical datasets, before using the corresponding models to generate the feature vector for the meta-classifier. In this case, the features represent the predictions obtained from the base classifiers. To realize the adoption of meta learning in the proposed invention, two challenges, including the varying-length feature vector, and the irrelevant knowledge problem, have to be overcome. To this end, the invention implements a zero-padding mechanism and a clustering approach to group the base classifiers in the knowledge extraction phase. 

# Setup
Clone with submodules

```shell script
git clone https://git.sagresearch.de/kompaki/cleanlearning/saged.git --recurse-submodules
```

Create a virtual environment and install requirements

 Update All Packages On Ubuntu
 ```shell script
 sudo apt update && sudo apt upgrade -y
 ```

```shell script
python3 -m venv venv 
source venv/bin/activate
pip3 install --upgrade setuptools
pip3 install --upgrade pip
```

Install error generator

```
cd baseline/setup/error_generator
python3 setup.py install 
```

Install pyTorch

```shell script
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

```shell script
pip3 install -e .
```

Install error detection and repair methods

## RAHA and BARAN

To install these methods, you can do so in two different ways:

Option 1: through pip3
```
pip3 install raha
```
Option 2: through the setup.py script which exists in the raha main directory
```
python3 setup.py install
```

## Katara

For this method, we do not need to install packages, but we need to download the knowledge base:
Download the knowledge base ([link](https://bit.ly/3hPkpWX)) and unzip the file. The files of the knowledge base should 
be placed in the following path.
```
cd detectors/katara/knowedge-base
``` 

## FAHES

To install FAHES, navigate to the src directory and run make to compile the source code
``` shell script
cd FAHES/src/
``` 

``` shell script
make clean && make
``` 

## HoloClean
To install HoloClean, read te installation part of its README file and make sure to active the Postgresql service.


# Usage

To run SAGED, you need to have enough classifiers trained. To train MLP classifiers for every dataset in the `datasets/` directory, run

```bash
python3 scripts/train_classifiers.py --datasets hospital --classifiers mlp_classifier
```

You can also specify datasets to train classifiers for with the `--datasets` argument and choose different classifiers.


## Run SAGED

The following command runs the SAGED detector to find errors in a dirty dataset using the models trained on historical data. The options are as follows:
    - --dirty-dataset: String denoting the name of the input dirty dataset
    - --historical-datasets: String denoting the names of the historical datasts used for training the base models
    - --tags: String used to describe the experiment
    - --features: String denoting which features to use while training the detection classifier. It can take two values, either `meta` to run SAGED on the meta features or `classic` to run a No-meta-learning detector (similar to RAHA and ED2).
    - --verbose: print the logs

```bash
python3 scripts/run_saged.py 
        --dirty-dataset hospital 
        --historical-datasets adult beers airbnb flights movies_1 
        --tags "excluding rayyan and tax" 
        --verbose
```

To track the experiments and log the metrics, parameters, artifacts, and models
```bash
mlflow ui
```

## Run baseline

The folloing command is used to detect errors in a dirty dataset using one of the baseline detectors.

```bash
python scripts/run_baseline.py --dirty-dataset nasa
                               --detection-method raha
                               --runs 10
                               --verbose

```

## Test E2E pipelines using baseline detectors

The following command runs a ML pipeline consists of an error detection, a repair method, and a ML model building module. 

```bash
python3 scripts/test_e2e_baseline.py 
        --dataset beers 
        --detection_method raha 
        --repair_method standardImputer
```

## Test E2E pipelines using SAGED

The following command runs a ML pipeline which uses SAGED for detecting errors. 

```bash
python3 scripts/test_e2e_saged.py 
        --dirty-dataset beers 
        --historical-datasets adult 
        --repair_method standardImputer
        --verbose
```

# Components

* Automatic featurization
* Base models selection: clustering, cosine similarity
* feature generation (serving base models)
* Labeling (sampling): random, active learning, clustering, or heuristic
* Label augmentation (sampling): none, random, active learning, prediction, knn-shapley


# Performance evaluation

## Experiments

* [ ] Ablation studies 
	- [ ] Which base models selection (similarity)
	- [ ] Which labeling strategy
	- [ ] Which label augmentation
        - [ ] test historical data
	
* [ ] Compare labeling budget of SAGED, ED2, and raha

* [ ] Detection accuracy of SAGED, ED2, raha, meta-datadriven, picket, holoclean, dBoost
* [ ] Detection runtime of SAGED, ED2, raha, meta-datadriven, picket, holoclean, dBoost

* [ ] E2E accuracy of SAGED, ED2, raha, meta-datadriven, picket, holoclean, dBoost (with GT as repair tool)
* [ ] E2E runtime of SAGED, ED2, raha, meta-datadriven, picket, holoclean, dBoost (with GT as repair tool)

* [ ] E2E accuracy of SAGED with activeclean, boostclean, cpclean (ML imputation as repair tool)
* [ ] E2E runtime of SAGED with activeclean, boostclean, cpclean (ML imputation as repair tool)

* [ ] scalability analysis (the amount of data)

* [ ] robustness analysis (the amount of errors)

* [ ] accuracy of SAGED when using base classifiers of a dataset to detect errors in the a dirty version of the same dataset (beers)

## Datasets 

* [ ] Beers
* [ ] Adult
* [ ] Breast cancer
* [ ] Rayyan
* [ ] Movies_1
* [ ] Nasa
* [ ] Soccer (scalability)
* [ ] Tax (scalability)
* [ ] Hospital
* [ ] Flights 
* [ ] Smart Factory 

# ToDos
* [x] Search for real datasets with real error profiles. 
* [x] write scripts to run the various experiments and plotting the results
* [x] Setup a server to run the experiments


# Follow-up Ideas

* Recognizing the error type, e.g., outlier, rule violation, missing value, etc.

* The current implementation of SAGED requires the availability of ground truth to label train data (y_train) of meta classifiers and to evaluate the performance (y_test). We need to modify this implementation to make it work even without ground truth. In this case, users will be asked to label the data directly. 