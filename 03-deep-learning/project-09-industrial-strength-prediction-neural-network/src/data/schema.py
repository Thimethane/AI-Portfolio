"""Dataset schema for the concrete compressive strength regression task."""

from __future__ import annotations

from dataclasses import dataclass


FEATURE_COLUMNS = [
    "cement",
    "blast_furnace_slag",
    "fly_ash",
    "water",
    "superplasticizer",
    "coarse_aggregate",
    "fine_aggregate",
    "age",
]

TARGET_COLUMN = "compressive_strength"
ALL_COLUMNS = [*FEATURE_COLUMNS, TARGET_COLUMN]


@dataclass(frozen=True)
class DatasetProfile:
    """Expected dataset shape and schema from the source project materials."""

    expected_rows: int = 1030
    expected_feature_count: int = 8
    expected_target_count: int = 1
    target_unit: str = "MPa"


DATASET_PROFILE = DatasetProfile()


def normalize_raw_column(column: str) -> str:
    """Map verbose UCI-style headers to stable production feature names."""

    normalized = " ".join(column.strip().lower().replace(",", "").split())

    if normalized.startswith("cement"):
        return "cement"
    if normalized.startswith("blast furnace slag"):
        return "blast_furnace_slag"
    if normalized.startswith("fly ash"):
        return "fly_ash"
    if normalized.startswith("water"):
        return "water"
    if normalized.startswith("superplasticizer"):
        return "superplasticizer"
    if normalized.startswith("coarse aggregate"):
        return "coarse_aggregate"
    if normalized.startswith("fine aggregate"):
        return "fine_aggregate"
    if normalized.startswith("age"):
        return "age"
    if normalized.startswith("concrete compressive strength"):
        return "compressive_strength"

    return normalized.replace(" ", "_")

