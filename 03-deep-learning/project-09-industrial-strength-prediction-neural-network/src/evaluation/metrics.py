"""Regression metrics for model validation and production reports."""

from __future__ import annotations

import math

import numpy as np
import torch


def _to_numpy(values) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=np.float64).reshape(-1)


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Compute MSE, RMSE, and MAE for a regression model."""

    actual = _to_numpy(y_true)
    predicted = _to_numpy(y_pred)
    residuals = actual - predicted
    mse = float(np.mean(residuals**2))
    rmse = float(math.sqrt(mse))
    mae = float(np.mean(np.abs(residuals)))
    return {"mse": mse, "rmse": rmse, "mae": mae}

