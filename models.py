from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def get_device() -> torch.device:
    """Return the best available torch device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model_and_tokenizer(model_name: str, num_labels: int):
    """Load a sequence classifier backbone and its tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )
    model.to(get_device())
    return model, tokenizer


class SentimentClassifier:
    """Small importable wrapper for macOS spawn / Apple Silicon workflows."""

    def __init__(self, model_name: str, num_labels: int):
        self.model_name = model_name
        self.num_labels = num_labels
        self.device = get_device()
        self.model, self.tokenizer = load_model_and_tokenizer(model_name, num_labels)

    def set_label_mapping(
        self,
        id2label: Dict[int, str],
        label2id: Dict[str, int],
    ) -> None:
        self.model.config.id2label = id2label.copy()
        self.model.config.label2id = label2id.copy()


def compute_metrics(
    eval_pred: tuple[Any, Any],
    negative_label: int = 0,
) -> Dict[str, float]:
    """Compute metrics with extra focus on the negative class."""
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]

    predictions = np.argmax(logits, axis=-1)
    negative_precision, negative_recall, negative_f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        labels=[negative_label],
        average=None,
        zero_division=0,
    )

    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
        "negative_precision": float(negative_precision[0]),
        "negative_recall": float(negative_recall[0]),
        "negative_f1": float(negative_f1[0]),
    }


def compute_confusion_matrix(
    true_labels: Iterable[int],
    predicted_labels: Iterable[int],
    label_order: Optional[Iterable[int]] = None,
) -> Dict[str, Any]:
    """Return the raw confusion matrix and the label order used."""
    true_labels = np.asarray(list(true_labels))
    predicted_labels = np.asarray(list(predicted_labels))

    if label_order is None:
        label_order = sorted(set(true_labels.tolist()) | set(predicted_labels.tolist()))

    label_order = list(label_order)
    matrix = confusion_matrix(true_labels, predicted_labels, labels=label_order)
    return {"matrix": matrix, "label_order": label_order}
