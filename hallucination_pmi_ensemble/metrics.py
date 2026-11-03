"""Metrics and confusion reporting for labelled hallucination predictions."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _binary_metrics(y_true: Sequence[int], y_pred: Sequence[int], scores: Sequence[float] | None = None) -> dict[str, Any]:
    labels = np.asarray(y_true, dtype=int)
    predictions = np.asarray(y_pred, dtype=int)
    result: dict[str, Any] = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
    }
    if scores is not None and len(np.unique(labels)) == 2:
        result["auroc"] = float(roc_auc_score(labels, scores))
        result["auprc"] = float(average_precision_score(labels, scores))
    else:
        result["auroc"] = None
        result["auprc"] = None
    return result


def evaluate_predictions(frame: pd.DataFrame) -> dict[str, Any]:
    """Return overall and per-dataset metrics when labels are available."""
    if "label" not in frame.columns:
        return {"rows": int(len(frame)), "labelled": False}
    required = {"ensemble_prediction", "ensemble_risk"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Prediction frame missing required columns: {sorted(missing)}")
    labels = frame["label"].astype(int)
    predictions = frame["ensemble_prediction"].astype(int)
    scores = frame["ensemble_risk"].astype(float)
    metrics: dict[str, Any] = {
        "rows": int(len(frame)),
        "labelled": True,
        "ensemble": _binary_metrics(labels, predictions, scores),
    }
    metrics["ensemble"]["classification_report"] = classification_report(
        labels,
        predictions,
        zero_division=0,
    )
    if "dataset_name" in frame.columns:
        metrics["per_dataset"] = {}
        for dataset_name, group in frame.groupby("dataset_name", sort=True):
            metrics["per_dataset"][str(dataset_name)] = {
                "rows": int(len(group)),
                "ensemble": _binary_metrics(
                    group["label"].astype(int),
                    group["ensemble_prediction"].astype(int),
                    group["ensemble_risk"].astype(float),
                ),
            }
    return metrics


def confusion_report(frame: pd.DataFrame) -> pd.DataFrame:
    if "label" not in frame.columns:
        return pd.DataFrame()
    groups = [("ALL", frame)]
    if "dataset_name" in frame.columns:
        groups.extend((str(name), group) for name, group in frame.groupby("dataset_name", sort=True))
    rows = []
    for name, group in groups:
        matrix = confusion_matrix(group["label"].astype(int), group["ensemble_prediction"].astype(int), labels=[0, 1])
        metrics = _binary_metrics(
            group["label"].astype(int),
            group["ensemble_prediction"].astype(int),
            group["ensemble_risk"].astype(float),
        )
        rows.append(
            {
                "dataset_name": name,
                "tn": int(matrix[0, 0]),
                "fp": int(matrix[0, 1]),
                "fn": int(matrix[1, 0]),
                "tp": int(matrix[1, 1]),
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "auroc": metrics["auroc"],
                "auprc": metrics["auprc"],
            }
        )
    return pd.DataFrame(rows)
