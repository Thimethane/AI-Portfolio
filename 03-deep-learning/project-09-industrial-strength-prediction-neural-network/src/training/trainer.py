"""Configurable PyTorch training loop for concrete strength regression."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch import nn

from data.schema import FEATURE_COLUMNS, TARGET_COLUMN
from evaluation.metrics import regression_metrics
from models.mlp import build_model


@dataclass
class TrainingConfig:
    epochs: int = 1200
    learning_rate: float = 0.001
    batch_size: int = 32
    optimizer: str = "adam"
    hidden_layers: tuple[int, ...] = (128, 64, 32)
    dropout: float = 0.0
    weight_decay: float = 0.0
    test_size: float = 0.2
    random_state: int = 42
    stratify_bins: int | None = 10
    device: str = "cpu"
    log_interval: int = 100
    use_mlflow: bool = False
    experiment_name: str = "industrial-strength-prediction"


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_optimizer(model: nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
    optimizer_name = config.optimizer.lower()
    if optimizer_name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    if optimizer_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    raise ValueError("optimizer must be one of: adam, sgd")


def train_model(prepared_data, config: TrainingConfig) -> dict[str, Any]:
    """Train the MLP with zero_grad, backward, and optimizer.step."""

    set_reproducibility(config.random_state)
    device = torch.device(config.device if torch.cuda.is_available() or config.device == "cpu" else "cpu")
    model_config = {
        "input_dim": len(FEATURE_COLUMNS),
        "hidden_layers": config.hidden_layers,
        "dropout": config.dropout,
    }
    model = build_model(model_config).to(device)
    criterion = nn.MSELoss()
    optimizer = create_optimizer(model, config)
    history: list[dict[str, float]] = []

    mlflow_run = None
    mlflow = None
    if config.use_mlflow:
        try:
            import mlflow as mlflow_module

            mlflow = mlflow_module
            mlflow.set_experiment(config.experiment_name)
            mlflow_run = mlflow.start_run()
            mlflow.log_params(_serializable_config(config))
        except Exception as exc:  # pragma: no cover - optional integration
            print(f"MLflow disabled because tracking could not start: {exc}")
            mlflow = None

    try:
        for epoch in range(1, config.epochs + 1):
            model.train()
            batch_losses = []
            for X_batch, y_batch in prepared_data.train_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad()
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                loss.backward()
                optimizer.step()
                batch_losses.append(float(loss.item()))

            if epoch == 1 or epoch % config.log_interval == 0 or epoch == config.epochs:
                metrics = evaluate_model(model, prepared_data, device)
                metrics["epoch"] = float(epoch)
                metrics["train_loss"] = float(np.mean(batch_losses))
                history.append(metrics)
                if mlflow:
                    mlflow.log_metrics(metrics, step=epoch)
    finally:
        if mlflow_run and mlflow:  # pragma: no cover - optional integration
            mlflow.end_run()

    final_metrics = evaluate_model(model, prepared_data, device)
    return {
        "model": model.cpu(),
        "model_config": model_config,
        "config": asdict(config),
        "history": history,
        "metrics": final_metrics,
    }


def evaluate_model(model: nn.Module, prepared_data, device: torch.device | str = "cpu") -> dict[str, float]:
    device = torch.device(device)
    model.eval()
    model.to(device)
    with torch.no_grad():
        train_predictions = model(prepared_data.X_train.to(device)).cpu()
        test_predictions = model(prepared_data.X_test.to(device)).cpu()

    train_metrics = regression_metrics(prepared_data.y_train, train_predictions)
    test_metrics = regression_metrics(prepared_data.y_test, test_predictions)
    return {
        "train_mse": train_metrics["mse"],
        "train_rmse": train_metrics["rmse"],
        "train_mae": train_metrics["mae"],
        "test_mse": test_metrics["mse"],
        "test_rmse": test_metrics["rmse"],
        "test_mae": test_metrics["mae"],
        "rmse_gap": abs(test_metrics["rmse"] - train_metrics["rmse"]),
    }


def save_training_artifact(path: str | Path, training_result: dict[str, Any], scaler) -> Path:
    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": training_result["model"].state_dict(),
            "model_config": training_result["model_config"],
            "training_config": training_result["config"],
            "feature_columns": FEATURE_COLUMNS,
            "target_column": TARGET_COLUMN,
            "scaler": scaler.to_dict(),
            "metrics": training_result["metrics"],
            "history": training_result["history"],
        },
        artifact_path,
    )
    return artifact_path


def _serializable_config(config: TrainingConfig) -> dict[str, object]:
    payload = asdict(config)
    payload["hidden_layers"] = ",".join(str(layer) for layer in config.hidden_layers)
    return payload
