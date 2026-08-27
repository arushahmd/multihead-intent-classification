"""Multi-head hierarchical intent-classification toolkit."""

from intent_classifier.data import (
    DatasetSplits,
    IntentRecord,
    LabelHierarchy,
    load_intent_csv,
    split_by_normalized_text,
)
from intent_classifier.evaluation import IntentMetrics, compute_metrics

__all__ = [
    "DatasetSplits",
    "IntentMetrics",
    "IntentRecord",
    "LabelHierarchy",
    "compute_metrics",
    "load_intent_csv",
    "split_by_normalized_text",
]

__version__ = "0.1.0"
