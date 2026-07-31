import torch

from evaluation.metrics import regression_metrics
from models.mlp import ConcreteStrengthMLP


def test_mlp_forward_returns_single_regression_output():
    model = ConcreteStrengthMLP(input_dim=8, hidden_layers=(16, 8))
    X = torch.randn(4, 8)

    y_pred = model(X)

    assert y_pred.shape == (4, 1)


def test_regression_metrics_are_computed_correctly():
    metrics = regression_metrics([3.0, 5.0], [2.0, 7.0])

    assert metrics["mse"] == 2.5
    assert round(metrics["rmse"], 6) == 1.581139
    assert metrics["mae"] == 1.5

