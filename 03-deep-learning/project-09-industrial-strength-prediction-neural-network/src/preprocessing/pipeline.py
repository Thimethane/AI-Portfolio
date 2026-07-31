"""Training data preparation pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from data.loader import split_features_target
from data.schema import FEATURE_COLUMNS
from preprocessing.standardizer import Standardizer, to_float_tensor_array


@dataclass
class PreparedData:
    train_loader: DataLoader
    X_train: torch.Tensor
    X_test: torch.Tensor
    y_train: torch.Tensor
    y_test: torch.Tensor
    scaler: Standardizer


def prepare_training_data(
    df,
    test_size: float = 0.2,
    random_state: int = 42,
    batch_size: int = 32,
    stratify_bins: int | None = None,
) -> PreparedData:
    """Create standardized tensors and an 80/20 train-test split."""

    features, target = split_features_target(df)
    stratify_labels = None
    if stratify_bins and stratify_bins > 1:
        stratify_labels = _make_regression_strata(target, stratify_bins)

    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_labels,
    )

    scaler = Standardizer(feature_columns=FEATURE_COLUMNS)
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    X_train = torch.tensor(to_float_tensor_array(X_train_scaled))
    X_test = torch.tensor(to_float_tensor_array(X_test_scaled))
    y_train = torch.tensor(to_float_tensor_array(y_train_raw).reshape(-1, 1))
    y_test = torch.tensor(to_float_tensor_array(y_test_raw).reshape(-1, 1))

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return PreparedData(
        train_loader=train_loader,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        scaler=scaler,
    )


def _make_regression_strata(target, bins: int):
    """Create quantile bins for regression stratification."""

    return target.rank(method="first").pipe(
        lambda ranks: pd.qcut(ranks, q=bins, labels=False, duplicates="drop")
    )
