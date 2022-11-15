"""Configuration enums and classes for SAGED."""

from dataclasses import dataclass
from enum import Enum

class ClassifierType(Enum):
    """Types of classifiers."""
    # TODO Add more classifiers
    MLP_CLASSIFIER = "mlp_classifier"

class ProfileType(Enum):
    """Types of profiles."""
    DISTRIBUTION = "distribution" # TODO Fix clustering with distribution
    STRUCTURE_FEATURES = "structure_features"

class ClusterAlgorithm(Enum):
    """Types of cluster algorithms"""
    KMEANS = "kmeans"
    WARD_AGGLOMERATIVE = "ward_agglomerative"

class LabelingMethod(Enum):
    """Labeling methods."""
    NONE = "none"
    CLUSTERING = "clustering"
    ACTIVE_LEARNING = "active_learning"

@dataclass
class Configuration():
    """SAGED configuration."""
    profile_type: ProfileType
    classifier_type: ClassifierType
    cluster_algorithm: ClusterAlgorithm
    labeling_method: LabelingMethod
    n_clusters: int
