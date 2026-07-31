"""Standardization logic shared by training and production inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd


@dataclass
class Standardizer:
    """Apply z-score standardization using training-set statistics."""

    feature_columns: list[str]
    mean_: dict[str, float] | None = None
    std_: dict[str, float] | None = None

    def fit(self, features: pd.DataFrame) -> "Standardizer":
        self._validate_columns(features)
        means = features[self.feature_columns].mean()
        stds = features[self.feature_columns].std(ddof=0).replace(0, 1.0)
        self.mean_ = means.astype(float).to_dict()
        self.std_ = stds.astype(float).to_dict()
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        self._validate_fitted()
        self._validate_columns(features)
        means = pd.Series(self.mean_)
        stds = pd.Series(self.std_)
        standardized = (features[self.feature_columns] - means) / stds
        return standardized.astype("float32")

    def fit_transform(self, features: pd.DataFrame) -> pd.DataFrame:
        return self.fit(features).transform(features)

    def to_dict(self) -> dict[str, object]:
        self._validate_fitted()
        return {
            "feature_columns": self.feature_columns,
            "mean": self.mean_,
            "std": self.std_,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "Standardizer":
        return cls(
            feature_columns=list(payload["feature_columns"]),
            mean_={key: float(value) for key, value in dict(payload["mean"]).items()},
            std_={key: float(value) for key, value in dict(payload["std"]).items()},
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Standardizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)

    def _validate_columns(self, features: pd.DataFrame) -> None:
        missing = [column for column in self.feature_columns if column not in features.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")

    def _validate_fitted(self) -> None:
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Standardizer must be fitted before transform")


def to_float_tensor_array(values: pd.DataFrame | pd.Series | np.ndarray) -> np.ndarray:
    """Convert tabular numeric values to a float32 NumPy array for tensor creation."""

    if isinstance(values, (pd.DataFrame, pd.Series)):
        values = values.to_numpy()
    return np.asarray(values, dtype=np.float32)

