from pathlib import Path

import torch

from data.loader import load_concrete_data
from data.schema import FEATURE_COLUMNS, TARGET_COLUMN
from preprocessing.pipeline import prepare_training_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Concrete_Data.csv"


def test_load_concrete_data_validates_source_schema():
    df = load_concrete_data(DATA_PATH)

    assert df.shape == (1030, 9)
    assert list(df.columns) == [*FEATURE_COLUMNS, TARGET_COLUMN]
    assert df.isna().sum().sum() == 0


def test_prepare_training_data_standardizes_training_features():
    df = load_concrete_data(DATA_PATH)
    prepared = prepare_training_data(
        df,
        test_size=0.2,
        random_state=42,
        batch_size=32,
        stratify_bins=10,
    )

    assert prepared.X_train.shape == (824, 8)
    assert prepared.X_test.shape == (206, 8)
    assert torch.allclose(prepared.X_train.mean(dim=0), torch.zeros(8), atol=1e-5)
    assert torch.allclose(prepared.X_train.std(dim=0, unbiased=False), torch.ones(8), atol=1e-5)
