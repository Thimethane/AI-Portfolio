"""Train the concrete strength MLP and save a production inference artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.loader import load_concrete_data
from preprocessing.pipeline import prepare_training_data
from training.trainer import TrainingConfig, save_training_artifact, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train concrete strength MLP")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--metrics-path", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default=None)
    parser.add_argument("--hidden-layers", default=None, help="Comma-separated layer sizes, e.g. 128,64,32")
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--stratify-bins", type=int, default=None)
    parser.add_argument("--use-mlflow", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = PROJECT_ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    data_path = Path(args.data_path or config["data"]["path"])
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path

    model_path = Path(args.model_path or config["artifacts"]["model_path"])
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path

    metrics_path = Path(args.metrics_path or config["artifacts"]["metrics_path"])
    if not metrics_path.is_absolute():
        metrics_path = PROJECT_ROOT / metrics_path

    training_settings = dict(config["training"])
    for key, value in {
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "optimizer": args.optimizer,
        "dropout": args.dropout,
        "weight_decay": args.weight_decay,
    }.items():
        if value is not None:
            training_settings[key] = value
    if args.hidden_layers:
        training_settings["hidden_layers"] = [
            int(layer.strip()) for layer in args.hidden_layers.split(",") if layer.strip()
        ]
    if args.use_mlflow:
        training_settings["use_mlflow"] = True
    training_settings["hidden_layers"] = tuple(training_settings["hidden_layers"])
    training_settings["test_size"] = float(config["data"]["test_size"])
    training_settings["random_state"] = int(config["data"]["random_state"])
    training_settings["stratify_bins"] = int(
        args.stratify_bins
        if args.stratify_bins is not None
        else config["data"].get("stratify_bins", 0)
    ) or None

    training_config = TrainingConfig(**training_settings)
    df = load_concrete_data(data_path)
    prepared_data = prepare_training_data(
        df,
        test_size=training_config.test_size,
        random_state=training_config.random_state,
        batch_size=training_config.batch_size,
        stratify_bins=training_config.stratify_bins,
    )
    result = train_model(prepared_data, training_config)
    artifact_path = save_training_artifact(model_path, result, prepared_data.scaler)

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(result["metrics"], indent=2), encoding="utf-8")

    print(f"Saved model artifact: {artifact_path}")
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
