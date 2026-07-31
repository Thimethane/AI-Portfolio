"""Load and validate the concrete compressive strength dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data.schema import ALL_COLUMNS, DATASET_PROFILE, FEATURE_COLUMNS, TARGET_COLUMN, normalize_raw_column


class DatasetValidationError(ValueError):
    """Raised when the dataset violates the project contract."""


def load_concrete_data(csv_path: str | Path) -> pd.DataFrame:
    """Load the raw CSV, normalize headers, and validate the regression schema."""

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    df = df.rename(columns={column: normalize_raw_column(column) for column in df.columns})
    df = df[ALL_COLUMNS]
    validate_concrete_data(df)
    return df


def validate_concrete_data(df: pd.DataFrame) -> None:
    """Validate shape, columns, numeric dtypes, and missing values."""

    missing_columns = [column for column in ALL_COLUMNS if column not in df.columns]
    if missing_columns:
        raise DatasetValidationError(f"Missing required columns: {missing_columns}")

    if len(df) != DATASET_PROFILE.expected_rows:
        raise DatasetValidationError(
            f"Expected {DATASET_PROFILE.expected_rows} observations, found {len(df)}"
        )

    if len(FEATURE_COLUMNS) != DATASET_PROFILE.expected_feature_count:
        raise DatasetValidationError("Feature schema does not match the source materials")

    if df[ALL_COLUMNS].isna().any().any():
        missing = df[ALL_COLUMNS].isna().sum()
        missing = missing[missing > 0].to_dict()
        raise DatasetValidationError(f"Dataset contains missing values: {missing}")

    non_numeric = [column for column in ALL_COLUMNS if not pd.api.types.is_numeric_dtype(df[column])]
    if non_numeric:
        raise DatasetValidationError(f"Columns must be numeric: {non_numeric}")

    if (df["age"] <= 0).any():
        raise DatasetValidationError("Age must be positive days")


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return feature matrix and target vector using the canonical schema."""

    validate_concrete_data(df)
    return df[FEATURE_COLUMNS].copy(), df[TARGET_COLUMN].copy()

