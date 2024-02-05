####################################################################
# SAGED: Implement a ML pipeline with SAGED used for error detection
# Authors: Mohamed Abdelaal
# Date: March 2023
# Software AG
# All Rights Reserved
####################################################################

from saged.saged import saged
from baseline.model.train import train_model
from baseline.setup.repairs.repair import repair, RepairMethod
from baseline.model.utils import ExperimentType, ExperimentName


def train_e2e_saged(clean_path: str,
                    dirty_path: str,
                    detections_path: str,
                    target_path: str,
                    repair_method: RepairMethod,
                    dirty_dataset: str,
                    historical_datasets: list,
                    features: str = 'meta',
                    profile: str = 'structure_features',
                    classifier: str = 'mlp_classifier',
                    clustering: str = 'kmeans',
                    propagate_labels: bool = False,
                    labeling: str = 'none',
                    similarity: str = 'clustering',
                    n_clusters: int = 1,
                    n_meta_features: int = 0,
                    label_augmentation: str = None,
                    exp_type: str = ExperimentName.MODELING.__str__(),
                    runs: int = 1,
                    num_labels: int = 20,
                    verbose=True,
                    tune_params=False,
                    epochs=500,
                    exp_name=ExperimentName.MODELING.__str__(),
                    error_rate=0):

    """
    Train a pipeline with error detector, data repair method, and keras models
    """

    # Detect errors
    if verbose:
        print("====================================================")
        print("=================== Error Detection ================")
        print("====================================================\n")
        
    try:
        list_precision, list_recall, list_f1_score, list_total_time = saged(dirty_dataset=dirty_dataset,
                                                                            historical_datasets=historical_datasets,
                                                                            features=features,
                                                                            profile=profile,
                                                                            classifier=classifier,
                                                                            clustering=clustering,
                                                                            propagate_labels=propagate_labels,
                                                                            labeling=labeling,
                                                                            similarity=similarity,
                                                                            n_clusters=n_clusters,
                                                                            n_meta_features=n_meta_features,
                                                                            label_augmentation=label_augmentation,
                                                                            runs=runs,
                                                                            exp_type=exp_type,
                                                                            num_labels=num_labels,
                                                                            verbose=verbose)
        print(f"Precision:{list_precision}, Recall:{list_recall}, F1 Score:{list_f1_score}, Total Time:{list_total_time}") 
    except Exception as e:
        print("[ERROR] Failed to run the error detection step")
        print("Exception: {}".format(e.args[0]))

    # Repair errors
    if verbose:
        print("====================================================")
        print("=================== Error Repair ===================")
        print("====================================================")
        print(f"Repairing the {dirty_dataset} dataset using the {repair_method.__str__()} repair method ...", end="", flush=True)
    
    repaired_df = repair(clean_path, dirty_path, target_path, detections_path, dirty_dataset, repair_method)
    
    if verbose:
        print("done.")
    
    # Train a model
    if verbose:
        print("====================================================")
        print("=================== Model Training =================")
        print("====================================================\n")

    exp_type = ExperimentType.E2E_PIPELINE.__str__() + '_' + 'saged' + '_' + repair_method.__str__()
    train_model(data_df=repaired_df,
                data_name=dirty_dataset,
                tune_params=tune_params,
                exp_type=exp_type,
                exp_name=exp_name,
                verbose=verbose,
                epochs=epochs,
                error_rate=error_rate)