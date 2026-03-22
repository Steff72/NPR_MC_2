import inspect
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed as hf_set_seed,
)


DEFAULT_ID2LABEL_3CLASS = {
    0: "negative",
    1: "neutral",
    2: "positive",
}
DEFAULT_ID2LABEL_2CLASS = {
    0: "negative",
    1: "positive",
}


def get_device() -> torch.device:
    """Return the best available torch device for this machine."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_global_seed(seed: int = 42) -> None:
    """Set random seeds across the common libraries used in the notebook."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    hf_set_seed(seed)


def _default_id2label(num_labels: Optional[int]) -> Dict[int, str]:
    if num_labels in (None, 3):
        return DEFAULT_ID2LABEL_3CLASS.copy()
    if num_labels == 2:
        return DEFAULT_ID2LABEL_2CLASS.copy()
    return {label_id: str(label_id) for label_id in range(num_labels)}


def resolve_label_mappings(
    num_labels: Optional[int] = None,
    id2label: Optional[Dict[int, str]] = None,
    label2id: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """Normalize and validate explicit label mappings."""
    if id2label is None and label2id is None:
        id2label = _default_id2label(num_labels)
    elif id2label is None:
        id2label = {int(label_id): str(label_name) for label_name, label_id in label2id.items()}
    else:
        id2label = {int(label_id): str(label_name) for label_id, label_name in id2label.items()}

    if label2id is None:
        label2id = {label_name: label_id for label_id, label_name in id2label.items()}
    else:
        label2id = {str(label_name): int(label_id) for label_name, label_id in label2id.items()}

    if num_labels is None:
        num_labels = len(id2label)

    expected_ids = set(range(num_labels))
    if set(id2label.keys()) != expected_ids:
        raise ValueError(
            f"id2label keys must match contiguous label ids {sorted(expected_ids)}; "
            f"received {sorted(id2label.keys())}."
        )
    if set(label2id.values()) != expected_ids:
        raise ValueError(
            f"label2id values must match contiguous label ids {sorted(expected_ids)}; "
            f"received {sorted(label2id.values())}."
        )

    return {
        "num_labels": num_labels,
        "id2label": id2label,
        "label2id": label2id,
    }


def load_tokenizer(model_name: str, **kwargs: Any):
    """Load a Hugging Face tokenizer for a classifier backbone."""
    return AutoTokenizer.from_pretrained(model_name, use_fast=True, **kwargs)


def load_sequence_classifier(
    model_name: str,
    num_labels: Optional[int] = None,
    id2label: Optional[Dict[int, str]] = None,
    label2id: Optional[Dict[str, int]] = None,
    **kwargs: Any,
):
    """Load a configurable sequence classification backbone with explicit labels."""
    label_config = resolve_label_mappings(
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=label_config["num_labels"],
        id2label=label_config["id2label"],
        label2id=label_config["label2id"],
        **kwargs,
    )


def compute_classification_metrics(
    true_labels: Iterable[int],
    predicted_labels: Iterable[int],
    negative_label: int = 0,
) -> Dict[str, float]:
    """Compute stable baseline metrics with explicit focus on the negative class."""
    true_array = np.asarray(list(true_labels))
    pred_array = np.asarray(list(predicted_labels))

    negative_precision, negative_recall, negative_f1, _ = precision_recall_fscore_support(
        true_array,
        pred_array,
        labels=[negative_label],
        average=None,
        zero_division=0,
    )

    return {
        "accuracy": accuracy_score(true_array, pred_array),
        "macro_f1": f1_score(true_array, pred_array, average="macro", zero_division=0),
        "negative_precision": float(negative_precision[0]),
        "negative_recall": float(negative_recall[0]),
        "negative_f1": float(negative_f1[0]),
    }


def compute_confusion_matrix(
    true_labels: Iterable[int],
    predicted_labels: Iterable[int],
    label_order: Optional[Iterable[int]] = None,
) -> Dict[str, Any]:
    """Return the raw confusion matrix plus the label order used to build it."""
    true_array = np.asarray(list(true_labels))
    pred_array = np.asarray(list(predicted_labels))
    if label_order is None:
        label_order = sorted(set(true_array.tolist()) | set(pred_array.tolist()))
    label_order = list(label_order)

    return {
        "matrix": confusion_matrix(true_array, pred_array, labels=label_order),
        "label_order": label_order,
    }


def _prepare_hf_dataset(
    df,
    text_column: str = "sentence",
    label_column: Optional[str] = "label",
):
    from datasets import Dataset

    required_columns = [text_column]
    if label_column is not None:
        required_columns.append(label_column)

    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    dataset_df = df[required_columns].copy().reset_index(drop=True)
    dataset = Dataset.from_pandas(dataset_df, preserve_index=False)
    if label_column is not None and label_column != "label":
        dataset = dataset.rename_column(label_column, "label")
    return dataset


def _tokenize_dataset(
    dataset,
    tokenizer,
    text_column: str = "sentence",
    max_length: Optional[int] = None,
):
    def tokenize_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
        tokenize_kwargs: Dict[str, Any] = {"truncation": True}
        if max_length is not None:
            tokenize_kwargs["max_length"] = max_length
        return tokenizer(batch[text_column], **tokenize_kwargs)

    return dataset.map(tokenize_batch, batched=True, remove_columns=[text_column])


def _build_training_arguments(
    output_dir: str,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    num_workers: int,
    seed: int,
    has_eval_dataset: bool,
) -> TrainingArguments:
    device = get_device()
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    training_kwargs: Dict[str, Any] = {
        "output_dir": output_dir,
        "learning_rate": learning_rate,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "num_train_epochs": epochs,
        "weight_decay": 0.01,
        "logging_dir": f"{output_dir}/logs",
        "logging_steps": 10,
        "report_to": "none",
        "seed": seed,
        "data_seed": seed,
        "dataloader_num_workers": num_workers,
        "dataloader_pin_memory": device.type == "cuda",
        "save_total_limit": 1,
    }

    if has_eval_dataset:
        training_kwargs["load_best_model_at_end"] = True
        training_kwargs["metric_for_best_model"] = "eval_negative_f1"
        training_kwargs["greater_is_better"] = True
    else:
        training_kwargs["load_best_model_at_end"] = False

    signature = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in signature.parameters:
        training_kwargs["eval_strategy"] = "epoch" if has_eval_dataset else "no"
        training_kwargs["save_strategy"] = "epoch" if has_eval_dataset else "no"
    else:
        training_kwargs["evaluation_strategy"] = "epoch" if has_eval_dataset else "no"
        training_kwargs["save_strategy"] = "epoch" if has_eval_dataset else "no"

    if "use_mps_device" in signature.parameters:
        training_kwargs["use_mps_device"] = device.type == "mps"
    if "use_cpu" in signature.parameters:
        training_kwargs["use_cpu"] = device.type == "cpu"
    elif "no_cuda" in signature.parameters:
        training_kwargs["no_cuda"] = device.type == "cpu"

    return TrainingArguments(**training_kwargs)


def _extract_labels(batch: Dict[str, Any]):
    if "labels" in batch:
        return batch.pop("labels")
    if "label" in batch:
        return batch.pop("label")
    return None


def build_trainer_metrics_fn(negative_label: int = 0):
    """Build a Trainer-compatible metric callback with stable output keys."""

    def compute_metrics(eval_pred) -> Dict[str, float]:
        logits, labels = eval_pred
        if isinstance(logits, tuple):
            logits = logits[0]
        predictions = np.argmax(logits, axis=-1)
        return compute_classification_metrics(
            true_labels=labels,
            predicted_labels=predictions,
            negative_label=negative_label,
        )

    return compute_metrics


def train_classifier(
    train_df,
    val_df=None,
    model_name: str = "distilbert-base-uncased",
    output_dir: str = "models",
    batch_size: int = 16,
    epochs: int = 3,
    learning_rate: float = 2e-5,
    num_workers: int = 0,
    seed: int = 42,
    negative_label: int = 0,
    text_column: str = "sentence",
    label_column: str = "label",
    max_length: Optional[int] = None,
    tokenizer=None,
    model=None,
    num_labels: Optional[int] = None,
    id2label: Optional[Dict[int, str]] = None,
    label2id: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """Train a configurable classifier backbone and return reusable artifacts."""
    set_global_seed(seed)
    label_config = resolve_label_mappings(
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    tokenizer = tokenizer or load_tokenizer(model_name)
    model = model or load_sequence_classifier(
        model_name=model_name,
        num_labels=label_config["num_labels"],
        id2label=label_config["id2label"],
        label2id=label_config["label2id"],
    )
    model.to(get_device())

    train_dataset = _tokenize_dataset(
        _prepare_hf_dataset(train_df, text_column=text_column, label_column=label_column),
        tokenizer=tokenizer,
        text_column=text_column,
        max_length=max_length,
    )

    eval_dataset = None
    if val_df is not None:
        eval_dataset = _tokenize_dataset(
            _prepare_hf_dataset(val_df, text_column=text_column, label_column=label_column),
            tokenizer=tokenizer,
            text_column=text_column,
            max_length=max_length,
        )

    trainer = Trainer(
        model=model,
        args=_build_training_arguments(
            output_dir=output_dir,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            num_workers=num_workers,
            seed=seed,
            has_eval_dataset=eval_dataset is not None,
        ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=build_trainer_metrics_fn(negative_label=negative_label),
    )
    trainer.train()

    return {
        "model_name": model_name,
        "model": trainer.model,
        "tokenizer": tokenizer,
        "trainer": trainer,
        "device": get_device(),
        "num_labels": label_config["num_labels"],
        "id2label": label_config["id2label"],
        "label2id": label_config["label2id"],
    }


def predict_classifier(
    model,
    tokenizer,
    predict_df,
    text_column: str = "sentence",
    batch_size: int = 32,
    max_length: Optional[int] = None,
) -> np.ndarray:
    """Run prediction for a sequence classifier and return hard labels."""
    encoded_dataset = _tokenize_dataset(
        _prepare_hf_dataset(predict_df, text_column=text_column, label_column=None),
        tokenizer=tokenizer,
        text_column=text_column,
        max_length=max_length,
    )
    data_loader = DataLoader(
        encoded_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    device = next(model.parameters()).device
    was_training = model.training
    model.eval()

    predictions = []
    with torch.no_grad():
        for batch in data_loader:
            _extract_labels(batch)
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch).logits
            predictions.extend(torch.argmax(logits, dim=-1).cpu().tolist())

    if was_training:
        model.train()

    return np.asarray(predictions)


def evaluate_classifier(
    model,
    tokenizer,
    eval_df,
    negative_label: int = 0,
    text_column: str = "sentence",
    label_column: str = "label",
    batch_size: int = 32,
    max_length: Optional[int] = None,
    include_confusion_matrix: bool = False,
    label_order: Optional[Iterable[int]] = None,
) -> Dict[str, Any]:
    """Evaluate a classifier on a labeled DataFrame using stable metric keys."""
    true_labels = eval_df[label_column].to_numpy()
    predicted_labels = predict_classifier(
        model=model,
        tokenizer=tokenizer,
        predict_df=eval_df[[text_column]].copy(),
        text_column=text_column,
        batch_size=batch_size,
        max_length=max_length,
    )

    metrics = compute_classification_metrics(
        true_labels=true_labels,
        predicted_labels=predicted_labels,
        negative_label=negative_label,
    )
    result: Dict[str, Any] = dict(metrics)
    if include_confusion_matrix:
        result["confusion"] = compute_confusion_matrix(
            true_labels=true_labels,
            predicted_labels=predicted_labels,
            label_order=label_order,
        )
    return result


class SentimentClassifier:
    """Compatibility wrapper around the generic baseline helper functions."""

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_labels: int = 3,
        output_dir: str = "models",
        negative_label: int = 0,
        id2label: Optional[Dict[int, str]] = None,
        label2id: Optional[Dict[str, int]] = None,
        seed: int = 42,
    ):
        label_config = resolve_label_mappings(
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
        )
        self.model_name = model_name
        self.output_dir = output_dir
        self.negative_label = negative_label
        self.seed = seed
        self.num_labels = label_config["num_labels"]
        self.id2label = label_config["id2label"]
        self.label2id = label_config["label2id"]
        self.tokenizer = load_tokenizer(model_name)
        self.model = load_sequence_classifier(
            model_name=model_name,
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id,
        )
        self.model.to(get_device())
        self.trainer = None

    def train(
        self,
        train_df,
        val_df,
        batch_size: int = 16,
        epochs: int = 3,
        learning_rate: float = 2e-5,
        num_workers: int = 0,
    ):
        trained = train_classifier(
            train_df=train_df,
            val_df=val_df,
            model_name=self.model_name,
            output_dir=self.output_dir,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            num_workers=num_workers,
            seed=self.seed,
            negative_label=self.negative_label,
            tokenizer=self.tokenizer,
            model=self.model,
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id,
        )
        self.model = trained["model"]
        self.tokenizer = trained["tokenizer"]
        self.trainer = trained["trainer"]
        return trained

    def predict(self, predict_df, batch_size: int = 32) -> np.ndarray:
        return predict_classifier(
            model=self.model,
            tokenizer=self.tokenizer,
            predict_df=predict_df,
            batch_size=batch_size,
        )

    def evaluate(
        self,
        eval_df,
        batch_size: int = 32,
        include_confusion_matrix: bool = False,
        label_order: Optional[Iterable[int]] = None,
    ) -> Dict[str, Any]:
        return evaluate_classifier(
            model=self.model,
            tokenizer=self.tokenizer,
            eval_df=eval_df,
            negative_label=self.negative_label,
            batch_size=batch_size,
            include_confusion_matrix=include_confusion_matrix,
            label_order=label_order,
        )

    def get_confusion_matrix(
        self,
        eval_df,
        batch_size: int = 32,
        label_order: Optional[Iterable[int]] = None,
    ) -> Dict[str, Any]:
        predicted_labels = self.predict(eval_df[["sentence"]].copy(), batch_size=batch_size)
        return compute_confusion_matrix(
            true_labels=eval_df["label"].to_numpy(),
            predicted_labels=predicted_labels,
            label_order=label_order,
        )

    def save_model(self, path: str) -> None:
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)


if __name__ == "__main__":
    print("Reusable classifier helpers loaded.")
