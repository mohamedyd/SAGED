"""Configuration enums and classes for SAGED."""

from dataclasses import dataclass
from enum import Enum

class Features(Enum):
    """Types of features."""
    META = "meta"
    CLASSIC = "classic"

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

class LabelingStrategy(Enum):
    """Labeling methods."""
    NONE = "none"
    CLUSTERING = "clustering"
    ACTIVE_LEARNING = "active_learning"
    HEURISTIC = "heuristic"

class Similarity(Enum):
    """Similarity measures"""
    CLUSTERING = "clustering"
    COSINE = "cosine"

@dataclass
class Configuration():
    """SAGED configuration."""
    features: Features
    profile_type: ProfileType
    classifier_type: ClassifierType
    cluster_algorithm: ClusterAlgorithm
    labeling_strategy: LabelingStrategy
    similarity: Similarity
    n_clusters: int
    n_meta_features: int
