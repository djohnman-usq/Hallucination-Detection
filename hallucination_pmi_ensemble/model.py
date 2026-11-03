"""Frozen PMI + length-control hallucination stacker.

This module intentionally contains only the deployable decision model. Upstream
detector feature generation is expensive and model-dependent; this package
expects those features to already exist in the input CSV.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StackerArtifact:
    features: tuple[str, ...]
    coefficients: dict[str, float]
    scaler_mean: dict[str, float]
    scaler_scale: dict[str, float]
    intercept: float
    threshold: float
    selected_c: float | None = None
    metadata: Mapping[str, Any] | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "StackerArtifact":
        features = tuple(str(item) for item in payload["features"])
        return cls(
            features=features,
            coefficients={name: float(payload["coefficients"][name]) for name in features},
            scaler_mean={name: float(payload["scaler_mean"][name]) for name in features},
            scaler_scale={name: float(payload["scaler_scale"][name]) for name in features},
            intercept=float(payload["intercept"]),
            threshold=float(payload["threshold"]),
            selected_c=float(payload["selected_c"]) if payload.get("selected_c") is not None else None,
            metadata={key: value for key, value in payload.items() if key not in {
                "features",
                "coefficients",
                "scaler_mean",
                "scaler_scale",
                "intercept",
                "threshold",
                "selected_c",
            }},
        )

    @classmethod
    def load(cls, path: str | Path) -> "StackerArtifact":
        return cls.from_mapping(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_mapping(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "features": list(self.features),
            "selected_c": self.selected_c,
            "threshold": self.threshold,
            "coefficients": self.coefficients,
            "scaler_mean": self.scaler_mean,
            "scaler_scale": self.scaler_scale,
            "intercept": self.intercept,
        }
        if self.metadata:
            result.update(dict(self.metadata))
        return result


class FrozenStacker:
    """Apply a validation-fitted logistic stacker to detector features."""

    def __init__(self, artifact: StackerArtifact):
        self.artifact = artifact

    @classmethod
    def load(cls, path: str | Path) -> "FrozenStacker":
        return cls(StackerArtifact.load(path))

    @property
    def features(self) -> tuple[str, ...]:
        return self.artifact.features

    def validate_frame(self, frame: pd.DataFrame) -> None:
        missing = [name for name in self.features if name not in frame.columns]
        if missing:
            raise ValueError(f"Input CSV is missing required model features: {missing}")

    def predict_risk(self, values: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(values, pd.DataFrame):
            self.validate_frame(values)
            matrix = values.loc[:, self.features].astype(float).to_numpy()
        else:
            matrix = np.asarray(values, dtype=float)
            if matrix.ndim != 2 or matrix.shape[1] != len(self.features):
                raise ValueError(
                    f"Expected a 2D array with {len(self.features)} feature columns; "
                    f"received shape {matrix.shape}."
                )
        mean = np.asarray([self.artifact.scaler_mean[name] for name in self.features], dtype=float)
        scale = np.asarray([self.artifact.scaler_scale[name] for name in self.features], dtype=float)
        coefficients = np.asarray([self.artifact.coefficients[name] for name in self.features], dtype=float)
        logits = ((matrix - mean) / np.clip(scale, 1e-12, None)) @ coefficients + self.artifact.intercept
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -500.0, 500.0)))

    def predict(self, values: pd.DataFrame | np.ndarray) -> np.ndarray:
        return (self.predict_risk(values) >= self.artifact.threshold).astype(int)

    def predict_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        risks = self.predict_risk(frame)
        output = frame.copy()
        output["stacking_risk"] = risks
        output["ensemble_risk"] = risks
        output["ensemble_prediction"] = (risks >= self.artifact.threshold).astype(int)
        output["ensemble_decision_source"] = "frozen_pmi_length_control_stacker"
        return output


def feature_matrix_from_rows(rows: Sequence[Mapping[str, Any]], features: Sequence[str]) -> np.ndarray:
    """Small convenience helper for non-CSV integrations."""
    return np.asarray([[float(row[name]) for name in features] for row in rows], dtype=float)
