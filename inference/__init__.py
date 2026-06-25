"""Allow `from inference import WitsGnnClassifier, route_decision`."""
from .gnn_inference import (
    AttentionGraphFeaturizer,
    WitsGnnClassifier,
    collapse_to_3class,
    route_decision,
    LABEL_NAMES,
    NODE_NAMES,
    DEFAULT_FEATURIZER,
    DEFAULT_MODEL_PT,
    DEFAULT_MODEL_METADATA,
)

__all__ = [
    "AttentionGraphFeaturizer",
    "WitsGnnClassifier",
    "collapse_to_3class",
    "route_decision",
    "LABEL_NAMES",
    "NODE_NAMES",
    "DEFAULT_FEATURIZER",
    "DEFAULT_MODEL_PT",
    "DEFAULT_MODEL_METADATA",
]
