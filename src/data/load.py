"""Loading and validation for the BRFSS 2015 Diabetes Health Indicators CSV.

Source: https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset
File used by this project: `diabetes_binary_health_indicators_BRFSS2015.csv`
(253,680 rows, 22 columns, ~86/14 class imbalance -- the *imbalanced* variant,
not the pre-balanced 50/50 split file).

Column names in the Kaggle CSV are CamelCase (`Diabetes_binary`, `HighBP`, ...).
We lowercase them on load so the rest of the project -- configs, notebooks, the
resampled artifact -- can use one consistent convention (`diabetes_binary`).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# The 21 feature columns plus the target, as published. Stored lowercased
# because `load_raw` normalises column names before validating.
EXPECTED_COLUMNS: tuple[str, ...] = (
    "diabetes_binary",
    "highbp",
    "highchol",
    "cholcheck",
    "bmi",
    "smoker",
    "stroke",
    "heartdiseaseorattack",
    "physactivity",
    "fruits",
    "veggies",
    "hvyalcoholconsump",
    "anyhealthcare",
    "nodocbccost",
    "genhlth",
    "menthlth",
    "physhlth",
    "diffwalk",
    "sex",
    "age",
    "education",
    "income",
)


class DataValidationError(ValueError):
    """Raised when the loaded CSV does not look like the expected dataset."""


def load_raw(
    input_path: str | Path,
    target_column: str = "diabetes_binary",
    *,
    normalize_columns: bool = True,
    strict_schema: bool = True,
) -> pd.DataFrame:
    """Read the raw CSV and validate it before anything downstream touches it.

    Args:
        input_path: path to the Kaggle CSV.
        target_column: name of the label column (post-normalisation).
        normalize_columns: lowercase and strip column names on load.
        strict_schema: require the exact BRFSS 2015 binary column set. Set
            False if you are deliberately feeding in a subset or a variant file.

    Raises:
        FileNotFoundError: if the CSV is not where the config says it is.
        DataValidationError: on schema, dtype, target or missing-value problems.
    """
    path = Path(input_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}. Download "
            "'diabetes_binary_health_indicators_BRFSS2015.csv' from Kaggle "
            "(alexteboul/diabetes-health-indicators-dataset) into data/raw/."
        )

    df = pd.read_csv(path)

    if normalize_columns:
        df.columns = [str(c).strip().lower() for c in df.columns]

    _validate(df, target_column, strict_schema=strict_schema)
    return df


def _validate(df: pd.DataFrame, target_column: str, *, strict_schema: bool) -> None:
    if df.empty:
        raise DataValidationError("Loaded dataframe is empty.")

    if target_column not in df.columns:
        raise DataValidationError(
            f"Target column {target_column!r} not found. Columns present: "
            f"{sorted(df.columns)}"
        )

    if strict_schema:
        expected = set(EXPECTED_COLUMNS)
        actual = set(df.columns)
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        if missing or unexpected:
            raise DataValidationError(
                "Schema mismatch against the BRFSS 2015 binary dataset.\n"
                f"  missing: {missing}\n"
                f"  unexpected: {unexpected}\n"
                "If this is intentional, set strict_schema: false in the config."
            )

    # Every column in this dataset is numeric (binary flags, ordinal scales, BMI).
    non_numeric = [
        c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])
    ]
    if non_numeric:
        raise DataValidationError(
            f"Expected all columns to be numeric; these are not: {non_numeric}"
        )

    n_missing = int(df.isna().sum().sum())
    if n_missing:
        worst = df.isna().sum()
        worst = worst[worst > 0].sort_values(ascending=False).to_dict()
        raise DataValidationError(
            f"Found {n_missing} missing values; the published dataset has none. "
            f"Per-column counts: {worst}"
        )

    labels = set(pd.unique(df[target_column]))
    if not labels <= {0, 0.0, 1, 1.0}:
        raise DataValidationError(
            f"Target {target_column!r} should be binary 0/1, saw: {sorted(labels)}. "
            "If you downloaded 'diabetes_012_...csv' by mistake, grab the "
            "'diabetes_binary_health_indicators_BRFSS2015.csv' file instead."
        )
    if len(labels) < 2:
        raise DataValidationError(
            f"Target {target_column!r} has a single class -- nothing to resample."
        )


def split_features_target(
    df: pd.DataFrame, target_column: str
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate X and y. The target is cast to int for clean class labels."""
    X = df.drop(columns=[target_column])
    y = df[target_column].astype(int)
    return X, y


def class_distribution(y: pd.Series, target_column: str = "target") -> pd.DataFrame:
    """Tidy count/percentage table for one label column -- used for W&B logging."""
    counts = y.value_counts().sort_index()
    total = int(counts.sum())
    return pd.DataFrame(
        {
            "class": [int(c) for c in counts.index],
            "count": [int(v) for v in counts.values],
            "percent": [round(100 * v / total, 4) for v in counts.values],
        }
    ).assign(target=target_column)
