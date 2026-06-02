from __future__ import annotations

import pandas as pd
from .data_loader import load_catch_data, load_luke_data


def _get_best_price_row(luke_df: pd.DataFrame, species_key: str, year: int) -> tuple[pd.Series | None, str | None]:
    species_df = luke_df[luke_df["species_key"].str.lower() == species_key.lower()].copy()

    if species_df.empty:
        return None, None

    species_df = species_df.dropna(subset=["price_eur_per_kg"]).copy()

    if species_df.empty:
        return None, None

    exact = species_df[species_df["year"] == year]
    if not exact.empty:
        return exact.iloc[0], "exact_year"

    species_df["year_distance"] = (species_df["year"] - year).abs()
    nearest = species_df.sort_values(["year_distance", "year"]).iloc[0]
    return nearest, "nearest_available_year"


def estimate_species_value(species_key: str, year: int) -> dict:
    catch_df = load_catch_data()
    luke_df = load_luke_data()

    catch_rows = catch_df[
        (catch_df["species_key"].str.lower() == species_key.lower()) &
        (catch_df["year"] == year)
    ]

    if catch_rows.empty:
        return {
            "species_key": species_key,
            "year": year,
            "message": f"No catch data found for species '{species_key}' in {year}."
        }

    catch_kg = float(catch_rows["catch_kg"].sum())

    price_row, method = _get_best_price_row(luke_df, species_key, year)

    if price_row is None:
        return {
            "species_key": species_key,
            "year": year,
            "catch_kg": catch_kg,
            "estimated_value_eur": None,
            "message": f"No market price found for species '{species_key}'."
        }

    price = float(price_row["price_eur_per_kg"])
    estimated_value = catch_kg * price

    return {
        "species_key": species_key,
        "year": year,
        "catch_kg": catch_kg,
        "price_eur_per_kg": price,
        "estimated_value_eur": estimated_value,
        "price_source_year": int(price_row["year"]),
        "price_method": method,
        "sources": ["catch_clean.csv", "luke_clean.csv"]
    }