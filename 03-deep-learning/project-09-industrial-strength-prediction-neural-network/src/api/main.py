"""FastAPI service for real-time concrete strength prediction."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from inference.predictor import ConcreteStrengthPredictor


DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "concrete_strength_mlp.pt"


class ConcreteMixRequest(BaseModel):
    cement: float = Field(..., ge=0)
    blast_furnace_slag: float = Field(..., ge=0)
    fly_ash: float = Field(..., ge=0)
    water: float = Field(..., ge=0)
    superplasticizer: float = Field(..., ge=0)
    coarse_aggregate: float = Field(..., ge=0)
    fine_aggregate: float = Field(..., ge=0)
    age: float = Field(..., gt=0)


class PredictionResponse(BaseModel):
    predicted_strength_mpa: float


def _artifact_path() -> Path:
    return Path(os.getenv("MODEL_ARTIFACT_PATH", str(DEFAULT_MODEL_PATH)))


def create_app() -> FastAPI:
    app = FastAPI(
        title="Industrial Strength Prediction Neural Network",
        version="1.0.0",
        description="PyTorch MLP regression service for concrete compressive strength.",
    )
    predictor_cache: dict[str, ConcreteStrengthPredictor] = {}

    def get_predictor() -> ConcreteStrengthPredictor:
        path = _artifact_path()
        cache_key = str(path.resolve())
        if cache_key not in predictor_cache:
            predictor_cache[cache_key] = ConcreteStrengthPredictor.from_artifact(path)
        return predictor_cache[cache_key]

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/model-info")
    def model_info() -> dict[str, object]:
        try:
            predictor = get_predictor()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {
            "model_type": "PyTorch MLP",
            "features": predictor.feature_columns,
            "metrics": predictor.metrics,
        }

    @app.post("/predict", response_model=PredictionResponse)
    def predict(payload: ConcreteMixRequest) -> PredictionResponse:
        try:
            predictor = get_predictor()
            payload_dict = payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()
            prediction = predictor.predict(payload_dict)[0]
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return PredictionResponse(predicted_strength_mpa=round(prediction, 3))

    return app


app = create_app()
