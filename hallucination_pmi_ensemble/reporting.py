"""Report-ready visualizations for the standalone stacker output."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


PMI_REPLACEMENT_SCORE_COLUMNS = {
    "Conditional PMI": "likelihood_risk",
    "Contradiction": "contradiction_risk",
    "FINCH": "finch_risk",
    "Ensemble": "ensemble_risk",
}


def _metric_row(labels: pd.Series, predictions: pd.Series, scores: pd.Series) -> dict[str, float]:
    labels = labels.astype(int)
    predictions = predictions.astype(int)
    values = {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
    }
    if labels.nunique() == 2:
        values["auroc"] = roc_auc_score(labels, scores.astype(float))
        values["auprc"] = average_precision_score(labels, scores.astype(float))
    else:
        values["auroc"] = np.nan
        values["auprc"] = np.nan
    return values


def detector_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if "label" not in frame.columns:
        return pd.DataFrame()
    groups = [("ALL", frame)]
    if "dataset_name" in frame.columns:
        groups.extend((str(name), group) for name, group in frame.groupby("dataset_name", sort=True))
    for dataset_name, group in groups:
        for detector, score_column in PMI_REPLACEMENT_SCORE_COLUMNS.items():
            if detector == "Ensemble":
                prediction_column = "ensemble_prediction"
            elif score_column == "likelihood_risk":
                prediction_column = "likelihood_vote"
            else:
                prediction_column = score_column.replace("_risk", "_vote")
            if score_column not in group.columns or prediction_column not in group.columns:
                continue
            rows.append(
                {
                    "dataset_name": dataset_name,
                    "detector": detector,
                    "rows": int(len(group)),
                    **_metric_row(group["label"], group[prediction_column], group[score_column]),
                }
            )
    summary = pd.DataFrame(rows)
    contexts = summary[summary["dataset_name"] != "ALL"]
    if not contexts.empty:
        metric_columns = ["accuracy", "precision", "recall", "f1", "auroc", "auprc"]
        macro = contexts.groupby("detector", as_index=False)[metric_columns].mean()
        macro.insert(0, "dataset_name", "MACRO")
        macro.insert(2, "rows", int(contexts.drop_duplicates("dataset_name")["rows"].sum()))
        summary = pd.concat([summary, macro], ignore_index=True)
    return summary


def save_ensemble_by_dataset(metrics: dict[str, Any], output: Path) -> None:
    rows = [("ALL", metrics["ensemble"])]
    rows.extend((name, item["ensemble"]) for name, item in metrics.get("per_dataset", {}).items())
    names = [name for name, _ in rows]
    metric_names = ["accuracy", "precision", "recall", "f1", "auroc", "auprc"]
    x = np.arange(len(names))
    width = 0.12
    figure, axis = plt.subplots(figsize=(12, 6.5))
    for index, metric in enumerate(metric_names):
        values = [result.get(metric, np.nan) for _, result in rows]
        axis.bar(x + (index - 2.5) * width, values, width, label=metric.upper())
    axis.set_title("Global Ensemble: Held-Out Performance by Dataset")
    axis.set_ylabel("Score")
    axis.set_xticks(x, names)
    axis.set_ylim(0, 1)
    axis.legend(ncol=3, frameon=False)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def save_detector_comparison(summary: pd.DataFrame, output: Path) -> None:
    overall = summary[summary["dataset_name"] == "ALL"].copy()
    metrics = ["accuracy", "f1", "auroc", "auprc"]
    x = np.arange(len(overall))
    width = 0.19
    figure, axis = plt.subplots(figsize=(11, 6.5))
    for index, metric in enumerate(metrics):
        values = overall[metric].to_numpy()
        bars = axis.bar(x + (index - 1.5) * width, values, width, label=metric.upper())
        for bar, value in zip(bars, values):
            axis.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.3f}", ha="center", fontsize=8)
    axis.set_title("Held-Out Detector Comparison (All Datasets)")
    axis.set_ylabel("Score")
    axis.set_xticks(x, overall["detector"])
    axis.set_ylim(0, 1)
    axis.legend(ncol=4, frameon=False)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def save_confusion_matrices(frame: pd.DataFrame, output: Path) -> None:
    groups = [("ALL", frame)]
    if "dataset_name" in frame.columns:
        groups.extend((str(name), group) for name, group in frame.groupby("dataset_name", sort=True))
    columns = 2
    rows = int(np.ceil(len(groups) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(10, 4.6 * rows))
    for axis, (name, group) in zip(np.ravel(axes), groups):
        matrix = confusion_matrix(group["label"], group["ensemble_prediction"], labels=[0, 1])
        axis.imshow(matrix, cmap="Purples")
        for (row, column), value in np.ndenumerate(matrix):
            axis.text(column, row, f"{value:,}", ha="center", va="center", fontsize=12)
        axis.set_title(name)
        axis.set_xticks([0, 1], ["Factual", "Hallucination"])
        axis.set_yticks([0, 1], ["Factual", "Hallucination"])
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
    for axis in np.ravel(axes)[len(groups) :]:
        axis.axis("off")
    figure.suptitle("Global Ensemble Confusion Matrices", y=0.98, fontsize=15)
    figure.subplots_adjust(top=0.90, hspace=0.38, wspace=0.28)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_score_distributions(frame: pd.DataFrame, output: Path) -> None:
    available = [(name, column) for name, column in PMI_REPLACEMENT_SCORE_COLUMNS.items() if column in frame.columns]
    figure, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 4.8), sharey=True)
    if len(available) == 1:
        axes = [axes]
    for axis, (detector, column) in zip(axes, available):
        values_all = frame[column].astype(float)
        score_min = float(values_all.min())
        score_max = float(values_all.max())
        bounded_risk = score_min >= 0.0 and score_max <= 1.0
        padding = max((score_max - score_min) * 0.04, 1e-6)
        bins = np.linspace(0, 1, 31) if bounded_risk else np.linspace(score_min - padding, score_max + padding, 31)
        for label, colour, name in [(0, "#457B9D", "Factual"), (1, "#E76F51", "Hallucination")]:
            values = frame.loc[frame["label"] == label, column].astype(float).to_numpy()
            axis.hist(values, bins=bins, alpha=0.62, density=True, color=colour, label=name)
        axis.set_title(detector)
        axis.set_xlabel("Hallucination risk" if bounded_risk else "Detector score (higher = hallucination)")
        if bounded_risk:
            axis.set_xlim(0, 1)
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Density")
    axes[-1].legend(frameon=False)
    figure.suptitle("Risk-Score Separation on Held-Out Data", y=1.02, fontsize=15)
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_f1_by_context(summary: pd.DataFrame, output: Path) -> None:
    contexts = summary[~summary["dataset_name"].isin(["ALL", "MACRO"])]
    if contexts.empty:
        return
    pivot = contexts.pivot(index="dataset_name", columns="detector", values="f1")
    figure, axis = plt.subplots(figsize=(11, 6.5))
    x = np.arange(len(pivot))
    width = 0.18
    for index, detector in enumerate(pivot.columns):
        values = pivot[detector].to_numpy()
        bars = axis.bar(x + (index - (len(pivot.columns) - 1) / 2) * width, values, width, label=detector)
        for bar, value in zip(bars, values):
            axis.text(bar.get_x() + bar.get_width() / 2, value + 0.012, f"{value:.3f}", ha="center", fontsize=8)
    axis.set_title("F1 by Evaluation Context (One Global Ensemble)")
    axis.set_ylabel("F1 score")
    axis.set_xticks(x, pivot.index)
    axis.set_ylim(0, 1)
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def create_report(frame: pd.DataFrame, metrics: dict[str, Any], output_dir: str | Path) -> pd.DataFrame:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    summary = detector_summary(frame)
    if summary.empty:
        return summary
    summary.to_csv(destination / "held_out_detector_summary.csv", index=False)
    save_ensemble_by_dataset(metrics, destination / "ensemble_per_dataset.png")
    save_detector_comparison(summary, destination / "detector_comparison.png")
    save_confusion_matrices(frame, destination / "confusion_matrices.png")
    save_score_distributions(frame, destination / "risk_score_distributions.png")
    save_f1_by_context(summary, destination / "f1_by_context.png")
    return summary
