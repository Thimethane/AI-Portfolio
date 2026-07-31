"""Run a single local prediction from a trained model artifact."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from inference.predictor import ConcreteStrengthPredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict concrete compressive strength")
    parser.add_argument("--model-path", default="models/concrete_strength_mlp.pt")
    parser.add_argument("--cement", type=float, required=True)
    parser.add_argument("--blast-furnace-slag", type=float, required=True)
    parser.add_argument("--fly-ash", type=float, required=True)
    parser.add_argument("--water", type=float, required=True)
    parser.add_argument("--superplasticizer", type=float, required=True)
    parser.add_argument("--coarse-aggregate", type=float, required=True)
    parser.add_argument("--fine-aggregate", type=float, required=True)
    parser.add_argument("--age", type=float, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path)
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path

    predictor = ConcreteStrengthPredictor.from_artifact(model_path)
    payload = {
        "cement": args.cement,
        "blast_furnace_slag": args.blast_furnace_slag,
        "fly_ash": args.fly_ash,
        "water": args.water,
        "superplasticizer": args.superplasticizer,
        "coarse_aggregate": args.coarse_aggregate,
        "fine_aggregate": args.fine_aggregate,
        "age": args.age,
    }
    prediction = predictor.predict(payload)[0]
    print(f"Predicted compressive strength: {prediction:.3f} MPa")


if __name__ == "__main__":
    main()

