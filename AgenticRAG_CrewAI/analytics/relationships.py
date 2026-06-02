from __future__ import annotations

from typing import Any

import pandas as pd

from .comparisons import get_combined_species_catch
from .data_loader import load_catch_data


def _load_water_quality_data() -> pd.DataFrame:
    from pathlib import Path

    base_dir = Path(__file__).resolve().parent.parent
    path = base_dir / "data" / "processed" / "water_quality_clean.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing processed file: {path}")
    return pd.read_csv(path)


def _normalize_species_key(species_key: str) -> str:
    return species_key.strip().lower()


def _normalize_metric_key(metric_key: str) -> str:
    return metric_key.strip()


def list_available_metrics() -> list[str]:
    df = _load_water_quality_data()
    return [col for col in df.columns if col != "year"]


def _interpret_correlation(value: float | None) -> str:
    if value is None:
        return "unavailable"

    abs_val = abs(value)
    direction = "positive" if value > 0 else "negative" if value < 0 else "neutral"

    if abs_val < 0.2:
        strength = "very weak"
    elif abs_val < 0.4:
        strength = "weak"
    elif abs_val < 0.6:
        strength = "moderate"
    elif abs_val < 0.8:
        strength = "strong"
    else:
        strength = "very strong"

    if value == 0:
        return "no linear relationship"

    return f"{strength} {direction} relationship"


def _build_species_series(species_key: str) -> pd.DataFrame:
    df = load_catch_data().copy()
    species_key = _normalize_species_key(species_key)

    sdf = (
        df[df["species_key"] == species_key]
        .groupby("year", dropna=True)["catch_kg"]
        .sum()
        .reset_index()
        .sort_values("year")
    )

    sdf = sdf.rename(columns={"catch_kg": "left_value"})
    return sdf


def _build_combined_species_series(species_keys: list[str]) -> pd.DataFrame:
    combined = get_combined_species_catch(species_keys)
    if "series" not in combined:
        return pd.DataFrame(columns=["year", "left_value"])
    return pd.DataFrame(combined["series"]).rename(columns={"catch_kg": "left_value"})


def _build_metric_series(metric_key: str) -> pd.DataFrame:
    water = _load_water_quality_data().copy()
    metric_key = _normalize_metric_key(metric_key)

    if metric_key not in water.columns:
        raise ValueError(f"Metric '{metric_key}' not found in water quality data.")

    mdf = water[["year", metric_key]].copy()
    mdf = mdf.rename(columns={metric_key: "right_value"})
    mdf = mdf.dropna(subset=["year"]).sort_values("year")
    return mdf


def _apply_lag_to_right_series(df: pd.DataFrame, lag_years: int) -> pd.DataFrame:
    """
    Positive lag_years means the right series is shifted forward in time.
    Example:
    previous year's temperature vs current catch
    -> lag_years=1
    -> temperature at year t is compared with catch at year t+1
    """
    shifted = df.copy()
    shifted["year"] = shifted["year"] + lag_years
    return shifted


def correlate_species_with_water_metric(
    species_key: str,
    metric_key: str,
    lag_years: int = 0,
) -> dict[str, Any]:
    left = _build_species_series(species_key)
    right = _build_metric_series(metric_key)
    right = _apply_lag_to_right_series(right, lag_years)

    merged = left.merge(right, on="year", how="inner").dropna(subset=["left_value", "right_value"])

    if len(merged) < 3:
        return {
            "species_key": species_key,
            "metric_key": metric_key,
            "lag_years": lag_years,
            "message": "Not enough overlapping years to compute correlation.",
            "source_left": "catch_clean.csv",
            "source_right": "water_quality_clean.csv",
        }

    corr = merged["left_value"].corr(merged["right_value"])

    return {
        "species_key": species_key,
        "metric_key": metric_key,
        "lag_years": lag_years,
        "n_years_used": int(len(merged)),
        "year_range": [int(merged["year"].min()), int(merged["year"].max())],
        "correlation": None if pd.isna(corr) else float(corr),
        "interpretation": _interpret_correlation(None if pd.isna(corr) else float(corr)),
        "data_points": merged.to_dict(orient="records"),
        "source_left": "catch_clean.csv",
        "source_right": "water_quality_clean.csv",
        "note": "Correlation does not imply causation.",
    }


def correlate_combined_species_with_water_metric(
    species_keys: list[str],
    metric_key: str,
    lag_years: int = 0,
) -> dict[str, Any]:
    left = _build_combined_species_series(species_keys)
    right = _build_metric_series(metric_key)
    right = _apply_lag_to_right_series(right, lag_years)

    merged = left.merge(right, on="year", how="inner").dropna(subset=["left_value", "right_value"])

    if len(merged) < 3:
        return {
            "species_keys": species_keys,
            "metric_key": metric_key,
            "lag_years": lag_years,
            "message": "Not enough overlapping years to compute correlation.",
            "source_left": "catch_clean.csv",
            "source_right": "water_quality_clean.csv",
        }

    corr = merged["left_value"].corr(merged["right_value"])

    return {
        "species_keys": species_keys,
        "metric_key": metric_key,
        "lag_years": lag_years,
        "n_years_used": int(len(merged)),
        "year_range": [int(merged["year"].min()), int(merged["year"].max())],
        "correlation": None if pd.isna(corr) else float(corr),
        "interpretation": _interpret_correlation(None if pd.isna(corr) else float(corr)),
        "data_points": merged.to_dict(orient="records"),
        "source_left": "catch_clean.csv",
        "source_right": "water_quality_clean.csv",
        "note": "Correlation does not imply causation.",
    }


def summarize_species_and_water_trends(
    species_key: str,
    metric_key: str,
) -> dict[str, Any]:
    left = _build_species_series(species_key)
    right = _build_metric_series(metric_key)

    merged = left.merge(right, on="year", how="inner").dropna(subset=["left_value", "right_value"])

    if len(merged) < 2:
        return {
            "species_key": species_key,
            "metric_key": metric_key,
            "message": "Not enough overlapping data to summarize trends.",
            "source_left": "catch_clean.csv",
            "source_right": "water_quality_clean.csv",
        }

    left_start = float(merged.iloc[0]["left_value"])
    left_end = float(merged.iloc[-1]["left_value"])
    right_start = float(merged.iloc[0]["right_value"])
    right_end = float(merged.iloc[-1]["right_value"])

    left_change = left_end - left_start
    right_change = right_end - right_start

    def describe(change: float) -> str:
        if change > 0:
            return "increase"
        if change < 0:
            return "decrease"
        return "no_change"

    return {
        "species_key": species_key,
        "metric_key": metric_key,
        "year_range": [int(merged["year"].min()), int(merged["year"].max())],
        "species_trend": describe(left_change),
        "metric_trend": describe(right_change),
        "species_change_absolute": left_change,
        "metric_change_absolute": right_change,
        "data_points": merged.to_dict(orient="records"),
        "source_left": "catch_clean.csv",
        "source_right": "water_quality_clean.csv",
    }