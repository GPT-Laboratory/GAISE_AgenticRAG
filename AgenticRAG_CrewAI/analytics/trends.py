from __future__ import annotations

import pandas as pd
from .data_loader import load_catch_data


def list_available_species() -> list[str]:
    df = load_catch_data()
    return sorted(df["species_key"].dropna().unique().tolist())


def get_total_catch_by_year() -> list[dict]:
    df = load_catch_data()

    result = (
        df.groupby("year", dropna=True)["catch_kg"]
        .sum()
        .reset_index()
        .sort_values("year")
    )

    return result.to_dict(orient="records")


def get_total_catch_for_year(year: int) -> dict:
    df = load_catch_data()
    year_df = df[df["year"] == year]

    if year_df.empty:
        return {
            "year": year,
            "total_catch_kg": None,
            "message": f"No catch data found for year {year}."
        }

    total = float(year_df["catch_kg"].sum())

    return {
        "year": year,
        "total_catch_kg": total,
        "source": "catch_clean.csv"
    }


def get_species_trend(species_key: str) -> dict:
    df = load_catch_data()
    species_df = df[df["species_key"].str.lower() == species_key.lower()].copy()

    if species_df.empty:
        return {
            "species_key": species_key,
            "message": f"No catch data found for species '{species_key}'."
        }

    species_df = species_df.sort_values("year")

    first_row = species_df.iloc[0]
    last_row = species_df.iloc[-1]

    start_value = float(first_row["catch_kg"])
    end_value = float(last_row["catch_kg"])
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
        "species_key": species_key,
        "start_year": int(first_row["year"]),
        "end_year": int(last_row["year"]),
        "start_value": start_value,
        "end_value": end_value,
        "absolute_change": absolute_change,
        "percent_change": percent_change,
        "trend": trend,
        "series": species_df[["year", "catch_kg"]].to_dict(orient="records"),
        "source": "catch_clean.csv"
    }


def get_top_species_by_year(year: int, limit: int = 5) -> dict:
    df = load_catch_data()
    year_df = df[df["year"] == year].copy()

    if year_df.empty:
        return {
            "year": year,
            "results": [],
            "message": f"No catch data found for year {year}."
        }

    ranked = (
        year_df.groupby("species_key", dropna=True)["catch_kg"]
        .sum()
        .sort_values(ascending=False)
        .head(limit)
        .reset_index()
    )

    return {
        "year": year,
        "results": ranked.to_dict(orient="records"),
        "source": "catch_clean.csv"
    }


def get_largest_species_change(direction: str = "increase") -> dict:
    df = load_catch_data()

    pivot = (
        df.pivot_table(index="year", columns="species_key", values="catch_kg", aggfunc="sum")
        .sort_index()
    )

    changes = []
    for species in pivot.columns:
        series = pivot[species].dropna()
        if len(series) < 2:
            continue

        first_value = float(series.iloc[0])
        last_value = float(series.iloc[-1])

        if first_value == 0:
            continue

        pct_change = ((last_value - first_value) / first_value) * 100.0

        changes.append({
            "species_key": species,
            "start_year": int(series.index[0]),
            "end_year": int(series.index[-1]),
            "start_value": first_value,
            "end_value": last_value,
            "percent_change": pct_change
        })

    if not changes:
        return {
            "message": "Not enough data to calculate species changes."
        }

    if direction == "decrease":
        best = min(changes, key=lambda x: x["percent_change"])
    else:
        best = max(changes, key=lambda x: x["percent_change"])

    best["direction"] = direction
    best["source"] = "catch_clean.csv"
    return best