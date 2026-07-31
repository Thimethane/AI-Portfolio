from pathlib import Path

import torch
from fastapi.testclient import TestClient

from api.main import create_app
from data.loader import load_concrete_data
from data.schema import FEATURE_COLUMNS, TARGET_COLUMN
from inference.predictor import ConcreteStrengthPredictor
from models.mlp import build_model
from preprocessing.standardizer import Standardizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Concrete_Data.csv"


def _sample_payload() -> dict[str, float]:
    return {
        "cement": 300.0,
        "blast_furnace_slag": 50.0,
        "fly_ash": 60.0,
        "water": 180.0,
        "superplasticizer": 6.0,
        "coarse_aggregate": 970.0,
        "fine_aggregate": 780.0,
        "age": 28.0,
    }


def _write_tiny_artifact(path: Path) -> None:
    df = load_concrete_data(DATA_PATH)
    scaler = Standardizer(FEATURE_COLUMNS).fit(df[FEATURE_COLUMNS])
    model_config = {"input_dim": 8, "hidden_layers": (8,), "dropout": 0.0}
    model = build_model(model_config)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model_config,
            "feature_columns": FEATURE_COLUMNS,
            "target_column": TARGET_COLUMN,
            "scaler": scaler.to_dict(),
            "metrics": {"test_rmse": 0.0},
            "history": [],
        },
        path,
    )


def test_predictor_loads_artifact_and_returns_float(tmp_path):
    artifact_path = tmp_path / "model.pt"
    _write_tiny_artifact(artifact_path)

    predictor = ConcreteStrengthPredictor.from_artifact(artifact_path)
    prediction = predictor.predict(_sample_payload())

    assert len(prediction) == 1
    assert isinstance(prediction[0], float)


def test_fastapi_predict_endpoint_uses_model_artifact(tmp_path, monkeypatch):
    artifact_path = tmp_path / "model.pt"
    _write_tiny_artifact(artifact_path)
    monkeypatch.setenv("MODEL_ARTIFACT_PATH", str(artifact_path))

    client = TestClient(create_app())
    response = client.post("/predict", json=_sample_payload())

    assert response.status_code == 200
    assert "predicted_strength_mpa" in response.json()

