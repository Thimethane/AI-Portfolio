"""Model artifact loading and prediction logic for production use."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import torch

from data.schema import FEATURE_COLUMNS
from models.mlp import build_model
from preprocessing.standardizer import Standardizer, to_float_tensor_array


class ConcreteStrengthPredictor:
    """Load a trained MLP artifact and produce MPa strength predictions."""

    def __init__(self, model, scaler: Standardizer, feature_columns: list[str], metrics: dict[str, float]):
        self.model = model
        self.scaler = scaler
        self.feature_columns = feature_columns
        self.metrics = metrics
        self.model.eval()

    @classmethod
    def from_artifact(cls, artifact_path: str | Path) -> "ConcreteStrengthPredictor":
        path = Path(artifact_path)
        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found: {path}")
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        model = build_model(artifact["model_config"])
        model.load_state_dict(artifact["model_state_dict"])
        scaler = Standardizer.from_dict(artifact["scaler"])
        return cls(
            model=model,
            scaler=scaler,
            feature_columns=list(artifact.get("feature_columns", FEATURE_COLUMNS)),
            metrics=dict(artifact.get("metrics", {})),
        )

    def predict(self, records: dict[str, float] | Iterable[dict[str, float]]) -> list[float]:
        if isinstance(records, dict):
            records = [records]
        frame = pd.DataFrame(records)
        missing = [column for column in self.feature_columns if column not in frame.columns]
        if missing:
            raise ValueError(f"Missing prediction features: {missing}")

        scaled = self.scaler.transform(frame[self.feature_columns])
        tensor = torch.tensor(to_float_tensor_array(scaled))
        with torch.no_grad():
            predictions = self.model(tensor).squeeze(1).numpy()
        return [float(value) for value in predictions]

