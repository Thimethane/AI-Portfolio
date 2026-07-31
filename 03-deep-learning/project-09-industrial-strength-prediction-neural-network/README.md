# Project 9: Industrial Strength Prediction Neural Network

Production-grade PyTorch regression system for predicting concrete compressive strength from material composition and curing age.

## Problem

Concrete compressive strength is a nonlinear function of cement, slag, fly ash, water, superplasticizer, coarse aggregate, fine aggregate, and age. This project converts the learning materials into a deployable Applied AI Engineering artifact with reproducible training, evaluation, inference, API serving, dashboarding, and containerization.

## Architecture

```text
src/
  data/            load and validate Concrete_Data.csv
  preprocessing/   train-fitted standardization and tensor preparation
  models/          PyTorch MLP built with nn.Module
  training/        MSE training loop with Adam/SGD and optional MLflow
  evaluation/      MSE, RMSE, MAE regression metrics
  inference/       artifact loading and prediction logic
  api/             FastAPI production service
dashboard/         Streamlit user interface
scripts/           training and local prediction CLIs
tests/             unit and integration tests
```

## Quick Start

```bash
cd 03-deep-learning/project-09-industrial-strength-prediction-neural-network
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts/train.py
pytest
```

## Train

```bash
python scripts/train.py --epochs 650 --learning-rate 0.01 --optimizer adam --hidden-layers 64,32 --dropout 0.05 --weight-decay 0.0001 --stratify-bins 10
```

The training artifact is saved to `models/concrete_strength_mlp.pt`, and metrics are written to `models/metrics.json`.

Use MLflow tracking when needed:

```bash
python scripts/train.py --use-mlflow
mlflow ui
```

## Predict Locally

```bash
python scripts/predict.py --cement 300 --blast-furnace-slag 50 --fly-ash 60 --water 180 --superplasticizer 6 --coarse-aggregate 970 --fine-aggregate 780 --age 28
```

## API

```bash
uvicorn api.main:app --app-dir src --reload
```

Example request:

```bash
curl -X POST http://127.0.0.1:8000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"cement\":300,\"blast_furnace_slag\":50,\"fly_ash\":60,\"water\":180,\"superplasticizer\":6,\"coarse_aggregate\":970,\"fine_aggregate\":780,\"age\":28}"
```

## Dashboard

```bash
streamlit run dashboard/streamlit_app.py
```

## Docker

```bash
docker build -t concrete-strength-api .
docker run -p 8000:8000 concrete-strength-api
```

## Evaluation

The training pipeline reports train and test MSE, RMSE, MAE, and the absolute train-test RMSE gap. A small gap indicates better generalization, while a large gap flags overfitting risk.

Current promoted artifact (`models/concrete_strength_mlp.pt`) was trained with the default config, including regression-stratified 80/20 splitting:

| Split | MSE | RMSE | MAE |
|---|---:|---:|---:|
| Train | 24.596 | 4.959 | 3.754 |
| Test | 30.541 | 5.526 | 4.086 |

Absolute RMSE gap: 0.567 MPa.

## Traceability

Source-derived requirements and production engineering enhancements are documented in `docs/engineering_spec.md`.
