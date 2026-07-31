# Engineering Specification Traceability

## Source-Derived Requirements

This implementation follows the provided concrete strength materials:

- Dataset: 1,030 observations, 8 quantitative inputs, 1 quantitative MPa target.
- Problem type: supervised regression in industrial AI and civil engineering.
- Data quality: complete numeric data with no missing values.
- Preprocessing: standardize input features to mean 0 and standard deviation 1.
- Model: Multi-Layer Perceptron implemented with PyTorch `nn.Module`.
- Algorithm: linear layers plus ReLU nonlinear activations.
- Loss: Mean Squared Error.
- Optimization: Adam by default, with SGD supported.
- Evaluation: MSE, RMSE, and MAE on both train and test sets.
- Split: 80/20 train-test split for generalization measurement.

## Engineering Enhancements

- Modular package layout under `src/` separates data, preprocessing, models, training, evaluation, inference, and API concerns.
- Regression-stratified target bins preserve the required 80/20 split while improving train/test distribution alignment.
- Saved model artifact includes model weights, architecture config, feature schema, scaler statistics, training config, history, and metrics.
- FastAPI exposes `/health`, `/model-info`, and `POST /predict` for production inference.
- Streamlit dashboard provides a user-facing prediction workflow over the same artifact used by the API.
- Dockerfile trains a reproducible model artifact during image build and serves the API with Uvicorn.
- Optional MLflow tracking, provided by `mlflow-skinny`, records hyperparameters and metrics when `--use-mlflow` is supplied.
- CLI-configurable dropout, hidden layer sizes, and optimizer weight decay support controlled regularization experiments.
- Pytest suite covers schema validation, standardization, model output shape, metrics, artifact loading, and API inference.

## Acceptance Notes

The implementation is production-oriented but intentionally compact. The code avoids notebook-only state, stores preprocessing parameters for deployment parity, and uses deterministic splits/seeds for repeatable results. Further production hardening could add authentication, model registry promotion, drift dashboards, CI deployment gates, and load testing.
