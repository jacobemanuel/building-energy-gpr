from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import (
    CATEGORICAL_FEATURES,
    FEATURES,
    RAW_CSV,
    RAW_XLSX,
    TARGETS,
    X6_LEVELS,
    X8_LEVELS,
)


def _normalize_column_name(name: str) -> str:
    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
    )


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip().replace(" ", "") for col in df.columns]
    return df


def _rename_columns_to_canonical(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mapping = {
        "relativecompactness": "X1",
        "surfacearea": "X2",
        "wallarea": "X3",
        "roofarea": "X4",
        "overallheight": "X5",
        "orientation": "X6",
        "glazingarea": "X7",
        "glazingareadistribution": "X8",
        "heatingload": "Y1",
        "coolingload": "Y2",
    }

    rename = {}
    for col in df.columns:
        key = _normalize_column_name(col)
        if key in mapping:
            rename[col] = mapping[key]

    if rename:
        df = df.rename(columns=rename)
    return df


def load_raw_data(path: Path | None = None) -> pd.DataFrame:
    if path is None:
        if RAW_XLSX.exists():
            path = RAW_XLSX
        # fall back to CSV if only an exported copy is available
        elif RAW_CSV.exists():
            path = RAW_CSV
        else:
            raise FileNotFoundError(
                "No raw data found. Expected ENB2012_data.xlsx or ENB2012_data.csv in data/raw."
            )

    path = Path(path)
    if path.suffix.lower() == ".xlsx":
        df = pd.read_excel(path, engine="openpyxl")
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    df = _standardize_columns(df)
    df = _rename_columns_to_canonical(df)
    return df


def coerce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in FEATURES + TARGETS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # keep the categorical columns as ints when that still works
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].round().astype("Int64")
    return df


def validate_data(df: pd.DataFrame) -> dict:
    missing_total = int(df.isna().sum().sum())
    missing_by_column = {k: int(v) for k, v in df.isna().sum().items()}
    duplicate_rows = int(df.duplicated().sum())

    outlier_counts = {}
    numeric_cols = [col for col in FEATURES + TARGETS if col in df.columns]
    for col in numeric_cols:
        series = df[col].dropna()
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = int(((series < lower) | (series > upper)).sum())
        outlier_counts[col] = outliers

    unique_counts = {col: int(df[col].nunique()) for col in df.columns}

    return {
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "missing_total": missing_total,
        "missing_by_column": missing_by_column,
        "duplicate_rows": duplicate_rows,
        "outlier_counts_iqr": outlier_counts,
        "unique_counts": unique_counts,
    }


def validate_enb2012_or_raise(df: pd.DataFrame) -> None:
    missing = [col for col in FEATURES + TARGETS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    subset = df[FEATURES + TARGETS]
    missing_by_column = subset.isna().sum()
    if int(missing_by_column.sum()) > 0:
        missing_cols = [col for col, count in missing_by_column.items() if int(count) > 0]
        raise ValueError(f"ENB2012 validation failed: missing values detected in {missing_cols}.")

    duplicate_rows = int(subset.duplicated().sum())
    if duplicate_rows > 0:
        raise ValueError(f"ENB2012 validation failed: found {duplicate_rows} duplicate rows.")

    allowed_levels = {"X6": set(X6_LEVELS), "X8": set(X8_LEVELS)}
    for col, allowed in allowed_levels.items():
        observed = set(pd.Series(subset[col]).dropna().astype(int).tolist())
        invalid = sorted(observed - allowed)
        if invalid:
            raise ValueError(
                f"ENB2012 validation failed: invalid levels for {col}: {invalid}. "
                f"Expected values within {sorted(allowed)}."
            )


def split_features_targets(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing = [col for col in FEATURES + TARGETS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    X = df[FEATURES].copy()
    y = df[TARGETS].copy()
    return X, y
