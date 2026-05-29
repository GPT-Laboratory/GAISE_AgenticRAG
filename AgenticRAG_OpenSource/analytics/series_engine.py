from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from analytics.data_loader import load_catch_data, load_luke_data


BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"


def load_water_quality_data() -> pd.DataFrame:
    path = PROCESSED_DIR / "water_quality_clean.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing processed file: {path}")
    return pd.read_csv(path)


def load_count_catch_data() -> pd.DataFrame:
    """
    Expected file format:
      year,item_key,count_units,unit

    Example rows:
      2010,signal_crayfish,462000,kpl
    """
    path = PROCESSED_DIR / "count_catch_clean.csv"
    if not path.exists():
        return pd.DataFrame(columns=["year", "item_key", "count_units", "unit"])
    return pd.read_csv(path)


def normalize_species_key(species_key: str) -> str:
    return species_key.strip().lower()


def get_available_species() -> list[str]:
    df = load_catch_data()
    return sorted(df["species_key"].dropna().astype(str).str.lower().unique().tolist())


def get_available_count_items() -> list[str]:
    df = load_count_catch_data()
    if df.empty:
        return []
    return sorted(df["item_key"].dropna().astype(str).str.lower().unique().tolist())


def get_available_metrics() -> list[str]:
    df = load_water_quality_data()
    return [col for col in df.columns if col != "year"]


def build_species_catch_series(
    species_key: str,
    start_year: int | None = None,
    end_year: int | None = None,
) -> pd.DataFrame:
    df = load_catch_data().copy()
    species_key = normalize_species_key(species_key)

    subset = df[df["species_key"].astype(str).str.lower() == species_key].copy()

    if start_year is not None:
        subset = subset[subset["year"] >= start_year]
    if end_year is not None:
        subset = subset[subset["year"] <= end_year]

    series = (
        subset.groupby("year", dropna=True)["catch_kg"]
        .sum()
        .reset_index()
        .sort_values("year")
        .rename(columns={"catch_kg": "value"})
    )

    return series


def build_total_catch_series(
    start_year: int | None = None,
    end_year: int | None = None,
) -> pd.DataFrame:
    df = load_catch_data().copy()

    if start_year is not None:
        df = df[df["year"] >= start_year]
    if end_year is not None:
        df = df[df["year"] <= end_year]

    series = (
        df.groupby("year", dropna=True)["catch_kg"]
        .sum()
        .reset_index()
        .sort_values("year")
        .rename(columns={"catch_kg": "value"})
    )

    return series


def build_combined_species_catch_series(
    species_keys: list[str],
    start_year: int | None = None,
    end_year: int | None = None,
) -> pd.DataFrame:
    normalized = [normalize_species_key(s) for s in species_keys if s]
    df = load_catch_data().copy()
    subset = df[df["species_key"].astype(str).str.lower().isin(normalized)].copy()

    if start_year is not None:
        subset = subset[subset["year"] >= start_year]
    if end_year is not None:
        subset = subset[subset["year"] <= end_year]

    series = (
        subset.groupby("year", dropna=True)["catch_kg"]
        .sum()
        .reset_index()
        .sort_values("year")
        .rename(columns={"catch_kg": "value"})
    )

    return series


def build_metric_series(
    metric_key: str,
    start_year: int | None = None,
    end_year: int | None = None,
) -> pd.DataFrame:
    df = load_water_quality_data().copy()

    if metric_key not in df.columns:
        raise ValueError(f"Metric '{metric_key}' not found in water_quality_clean.csv")

    subset = df[["year", metric_key]].copy()

    if start_year is not None:
        subset = subset[subset["year"] >= start_year]
    if end_year is not None:
        subset = subset[subset["year"] <= end_year]

    subset = (
        subset.dropna(subset=["year"])
        .sort_values("year")
        .rename(columns={metric_key: "value"})
    )

    return subset


def interpolate_species_prices(luke_df: pd.DataFrame) -> pd.DataFrame:
    df = luke_df.copy()
    if "species_key" not in df.columns:
        return df

    df["species_key"] = df["species_key"].astype(str).str.lower()
    df = df.sort_values(["species_key", "year"]).reset_index(drop=True)

    if "price_eur_per_kg" in df.columns:
        df["price_eur_per_kg_filled"] = (
            df.groupby("species_key")["price_eur_per_kg"]
            .transform(lambda s: s.interpolate(limit_direction="both"))
        )

    return df


def interpolate_count_item_prices(count_price_df: pd.DataFrame) -> pd.DataFrame:
    df = count_price_df.copy()
    if df.empty:
        return df

    df["item_key"] = df["item_key"].astype(str).str.lower()
    df = df.sort_values(["item_key", "year"]).reset_index(drop=True)

    df["price_eur_per_unit_filled"] = (
        df.groupby("item_key")["price_eur_per_unit"]
        .transform(lambda s: s.interpolate(limit_direction="both"))
    )

    return df


def build_species_value_series(
    species_key: str,
    start_year: int | None = None,
    end_year: int | None = None,
) -> pd.DataFrame:
    species_key = normalize_species_key(species_key)

    catch_df = load_catch_data().copy()
    luke_df = interpolate_species_prices(load_luke_data())

    catch_subset = catch_df[catch_df["species_key"].astype(str).str.lower() == species_key].copy()

    if start_year is not None:
        catch_subset = catch_subset[catch_subset["year"] >= start_year]
    if end_year is not None:
        catch_subset = catch_subset[catch_subset["year"] <= end_year]

    catch_series = (
        catch_subset.groupby("year", dropna=True)["catch_kg"]
        .sum()
        .reset_index()
        .sort_values("year")
    )

    price_series = (
        luke_df[luke_df["species_key"] == species_key][["year", "price_eur_per_kg_filled"]]
        .drop_duplicates()
        .sort_values("year")
    )

    merged = catch_series.merge(price_series, on="year", how="left")
    merged["estimated_value_eur"] = merged["catch_kg"] * merged["price_eur_per_kg_filled"]

    return merged.rename(columns={"estimated_value_eur": "value"})


def build_count_price_table() -> pd.DataFrame:
    """
    Builds a generic count-based price table from Luke data.

    Expected Luke columns:
      year, species_key, quantity_units, value_eur

    We compute:
      price_eur_per_unit = value_eur / quantity_units
    """
    luke_df = load_luke_data().copy()

    required = {"year", "species_key", "value_eur"}
    if not required.issubset(luke_df.columns):
        return pd.DataFrame(columns=["year", "item_key", "price_eur_per_unit"])

    quantity_col = None
    for candidate in ["count_units", "quantity_units", "count_kpl", "quantity_kpl"]:
        if candidate in luke_df.columns:
            quantity_col = candidate
            break

    if quantity_col is None:
        return pd.DataFrame(columns=["year", "item_key", "price_eur_per_unit"])

    subset = luke_df[["year", "species_key", quantity_col, "value_eur"]].copy()
    subset = subset.rename(columns={"species_key": "item_key", quantity_col: "quantity_units"})
    subset["item_key"] = subset["item_key"].astype(str).str.lower()
    subset["price_eur_per_unit"] = subset["value_eur"] / subset["quantity_units"].replace(0, pd.NA)

    return subset[["year", "item_key", "price_eur_per_unit"]]


def build_count_based_value_series(
    item_key: str,
    start_year: int | None = None,
    end_year: int | None = None,
) -> pd.DataFrame:
    """
    Generic count-based valuation series.

    Local data:
      count_catch_clean.csv -> year,item_key,count_units
    Market data:
      Luke-derived count prices -> year,item_key,price_eur_per_unit
    """
    item_key = normalize_species_key(item_key)

    catch_df = load_count_catch_data().copy()
    if catch_df.empty:
        return pd.DataFrame(columns=["year", "count_units", "price_eur_per_unit_filled", "value"])

    catch_subset = catch_df[catch_df["item_key"].astype(str).str.lower() == item_key].copy()

    if start_year is not None:
        catch_subset = catch_subset[catch_subset["year"] >= start_year]
    if end_year is not None:
        catch_subset = catch_subset[catch_subset["year"] <= end_year]

    catch_series = (
        catch_subset.groupby("year", dropna=True)["count_units"]
        .sum()
        .reset_index()
        .sort_values("year")
    )

    price_table = interpolate_count_item_prices(build_count_price_table())
    if not price_table.empty:
        price_series = (
            price_table[price_table["item_key"] == item_key][["year", "price_eur_per_unit_filled"]]
            .drop_duplicates()
            .sort_values("year")
        )
    else:
        price_series = pd.DataFrame(columns=["year", "price_eur_per_unit_filled"])

    merged = catch_series.merge(price_series, on="year", how="left")
    merged["value"] = merged["count_units"] * merged["price_eur_per_unit_filled"]

    return merged


def build_all_species_value_table(
    year: int | None = None,
) -> pd.DataFrame:
    catch_df = load_catch_data().copy()
    luke_df = interpolate_species_prices(load_luke_data())

    if year is not None:
        catch_df = catch_df[catch_df["year"] == year]

    catch_grouped = (
        catch_df.groupby(["year", "species_key"], dropna=True)["catch_kg"]
        .sum()
        .reset_index()
    )

    price_df = luke_df[["year", "species_key", "price_eur_per_kg_filled"]].copy()

    merged = catch_grouped.merge(
        price_df,
        on=["year", "species_key"],
        how="left",
    )

    merged["estimated_value_eur"] = merged["catch_kg"] * merged["price_eur_per_kg_filled"]
    return merged.sort_values(["year", "estimated_value_eur"], ascending=[True, False]).reset_index(drop=True)


def shift_series_years(series_df: pd.DataFrame, lag_years: int) -> pd.DataFrame:
    shifted = series_df.copy()
    shifted["year"] = shifted["year"] + lag_years
    return shifted


def align_two_series(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    right_lag_years: int = 0,
) -> pd.DataFrame:
    left = left_df[["year", "value"]].copy().rename(columns={"value": "left_value"})
    right = right_df[["year", "value"]].copy().rename(columns={"value": "right_value"})
    right = shift_series_years(right, right_lag_years)

    merged = left.merge(right, on="year", how="inner")
    merged = merged.dropna(subset=["left_value", "right_value"]).sort_values("year")
    return merged


def summarize_series(series_df: pd.DataFrame) -> dict[str, Any]:
    if series_df.empty:
        return {"message": "Series is empty."}

    sdf = series_df.sort_values("year").reset_index(drop=True)
    first = sdf.iloc[0]
    last = sdf.iloc[-1]

    start_value = float(first["value"])
    end_value = float(last["value"])
    absolute_change = end_value - start_value

    percent_change = None
    if start_value != 0:
        percent_change = (absolute_change / start_value) * 100.0

    trend = "no_change"
    if absolute_change > 0:
        trend = "increase"
    elif absolute_change < 0:
        trend = "decrease"

    return {
        "start_year": int(first["year"]),
        "end_year": int(last["year"]),
        "start_value": start_value,
        "end_value": end_value,
        "absolute_change": absolute_change,
        "percent_change": percent_change,
        "trend": trend,
        "series": sdf.to_dict(orient="records"),
    }


def correlate_aligned_series(aligned_df: pd.DataFrame) -> dict[str, Any]:
    if len(aligned_df) < 3:
        return {
            "message": "Not enough overlapping years to compute correlation."
        }

    corr = aligned_df["left_value"].corr(aligned_df["right_value"])

    if pd.isna(corr):
        corr_value = None
        interpretation = "unavailable"
    else:
        corr_value = float(corr)
        abs_val = abs(corr_value)
        direction = "positive" if corr_value > 0 else "negative" if corr_value < 0 else "neutral"

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

        interpretation = "no linear relationship" if corr_value == 0 else f"{strength} {direction} relationship"

    return {
        "n_years_used": int(len(aligned_df)),
        "year_range": [int(aligned_df["year"].min()), int(aligned_df["year"].max())],
        "correlation": corr_value,
        "interpretation": interpretation,
        "data_points": aligned_df.to_dict(orient="records"),
        "note": "Correlation does not imply causation.",
    }


def compare_two_series(series_a: pd.DataFrame, series_b: pd.DataFrame) -> dict[str, Any]:
    summary_a = summarize_series(series_a)
    summary_b = summarize_series(series_b)

    if "message" in summary_a or "message" in summary_b:
        return {
            "series_a": summary_a,
            "series_b": summary_b,
            "message": "Unable to compare one or both series."
        }

    a_change = summary_a.get("percent_change")
    b_change = summary_b.get("percent_change")

    faster_growth = None
    if a_change is not None and b_change is not None:
        if a_change > b_change:
            faster_growth = "left"
        elif b_change > a_change:
            faster_growth = "right"
        else:
            faster_growth = "tie"

    return {
        "series_a": summary_a,
        "series_b": summary_b,
        "faster_growth": faster_growth,
    }


def find_crossover_year(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
) -> dict[str, Any]:
    left = left_df[["year", "value"]].copy().rename(columns={"value": "left_value"})
    right = right_df[["year", "value"]].copy().rename(columns={"value": "right_value"})

    merged = left.merge(right, on="year", how="inner").dropna().sort_values("year")
    if merged.empty:
        return {"message": "No overlapping years available for crossover analysis."}

    merged["difference"] = merged["left_value"] - merged["right_value"]

    crossover_year = None
    for i in range(1, len(merged)):
        prev_diff = merged.iloc[i - 1]["difference"]
        curr_diff = merged.iloc[i]["difference"]

        if prev_diff == 0:
            crossover_year = int(merged.iloc[i - 1]["year"])
            break

        if (prev_diff < 0 < curr_diff) or (prev_diff > 0 > curr_diff):
            crossover_year = int(merged.iloc[i]["year"])
            break

    return {
        "crossover_year": crossover_year,
        "data_points": merged.to_dict(orient="records"),
    }