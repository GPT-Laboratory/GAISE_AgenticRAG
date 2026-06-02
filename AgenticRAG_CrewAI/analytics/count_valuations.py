from __future__ import annotations

from typing import Any

import pandas as pd

from .economic_rankings import get_top_value_species_by_year
from .series_engine import (
    build_count_based_value_series,
    find_crossover_year,
    summarize_series,
)


def summarize_count_item_value_trend(item_key: str) -> dict[str, Any]:
    series = build_count_based_value_series(item_key)
    summary = summarize_series(series)

    return {
        "item_key": item_key,
        **summary,
        "source": ["count_catch_clean.csv", "luke_clean.csv"],
    }


def compare_count_item_to_top_fish_by_year(item_key: str, year: int) -> dict[str, Any]:
    count_series = build_count_based_value_series(item_key)
    count_year = count_series[count_series["year"] == year].copy()

    top_fish = get_top_value_species_by_year(year=year, limit=1)

    if count_year.empty:
        return {
            "item_key": item_key,
            "year": year,
            "message": f"No count-based local catch data found for '{item_key}' in {year}.",
            "reason": "The processed count-based local catch series has no row for that item and year.",
            "source": ["count_catch_clean.csv", "luke_clean.csv"],
        }

    if not top_fish.get("results"):
        return {
            "item_key": item_key,
            "year": year,
            "message": f"No top fish value result found for {year}.",
            "reason": "The fish valuation table did not produce a ranked value result for that year.",
            "source": ["catch_clean.csv", "luke_clean.csv"],
        }

    item_value = count_year.iloc[0]["value"]
    top_row = top_fish["results"][0]
    top_species = top_row["species_key"]
    top_value = top_row["estimated_value_eur"]

    winner = None
    if pd.notna(item_value) and pd.notna(top_value):
        if float(item_value) > float(top_value):
            winner = item_key
        elif float(top_value) > float(item_value):
            winner = top_species
        else:
            winner = "tie"

    return {
        "item_key": item_key,
        "year": year,
        "item_value_eur": None if pd.isna(item_value) else float(item_value),
        "top_fish_species_key": top_species,
        "top_fish_value_eur": None if pd.isna(top_value) else float(top_value),
        "winner": winner,
        "source": ["count_catch_clean.csv", "catch_clean.csv", "luke_clean.csv"],
    }


def compare_count_item_to_top_fish_over_time(item_key: str) -> dict[str, Any]:
    count_series = build_count_based_value_series(item_key)

    if count_series.empty:
        return {
            "item_key": item_key,
            "message": f"No count-based local catch value series found for '{item_key}'.",
            "reason": "The processed count-based series is missing or empty for that item.",
            "source": ["count_catch_clean.csv", "luke_clean.csv"],
        }

    from .economic_rankings import get_yearly_top_value_species

    top_fish = get_yearly_top_value_species()
    top_rows = top_fish.get("results", [])

    if not top_rows:
        return {
            "item_key": item_key,
            "message": "No yearly top fish value series found.",
            "reason": "The fish valuation pipeline did not produce yearly leaders.",
            "source": ["catch_clean.csv", "luke_clean.csv"],
        }

    top_df = pd.DataFrame(top_rows).rename(
        columns={
            "estimated_value_eur": "top_fish_value_eur",
            "species_key": "top_fish_species_key",
        }
    )

    merged = count_series.merge(top_df, on="year", how="inner")

    if merged.empty:
        return {
            "item_key": item_key,
            "message": "No overlapping years between count-based item values and top fish values.",
            "reason": "The two yearly series do not overlap with usable values.",
            "source": ["count_catch_clean.csv", "catch_clean.csv", "luke_clean.csv"],
        }

    merged["item_higher"] = merged["value"] > merged["top_fish_value_eur"]

    higher_years = merged.loc[merged["item_higher"] == True, "year"].tolist()
    first_higher_year = int(min(higher_years)) if higher_years else None

    return {
        "item_key": item_key,
        "ever_higher": len(higher_years) > 0,
        "first_higher_year": first_higher_year,
        "higher_years": [int(y) for y in higher_years],
        "data_points": merged.to_dict(orient="records"),
        "source": ["count_catch_clean.csv", "catch_clean.csv", "luke_clean.csv"],
    }


def find_count_item_vs_top_fish_crossover(item_key: str) -> dict[str, Any]:
    count_series = build_count_based_value_series(item_key)

    if count_series.empty:
        return {
            "item_key": item_key,
            "message": f"No count-based local catch value series found for '{item_key}'.",
            "reason": "The processed count-based series is missing or empty for that item.",
            "source": ["count_catch_clean.csv", "luke_clean.csv"],
        }

    from .economic_rankings import get_yearly_top_value_species

    top_fish = get_yearly_top_value_species()
    yearly_top_rows = []
    for row in top_fish.get("results", []):
        yearly_top_rows.append(
            {
                "year": row["year"],
                "value": row["estimated_value_eur"],
                "top_species_key": row["species_key"],
            }
        )

    if not yearly_top_rows:
        return {
            "item_key": item_key,
            "message": "No yearly top fish value series found.",
            "reason": "The fish valuation pipeline did not produce yearly leaders.",
            "source": ["catch_clean.csv", "luke_clean.csv"],
        }

    top_df = pd.DataFrame(yearly_top_rows)
    merged = (
        count_series[["year", "value"]]
        .rename(columns={"value": "left_value"})
        .merge(
            top_df[["year", "value"]].rename(columns={"value": "right_value"}),
            on="year",
            how="inner",
        )
        .dropna()
        .sort_values("year")
    )

    if merged.empty:
        return {
            "item_key": item_key,
            "message": "No overlapping years available for crossover analysis.",
            "reason": "The count-based value series and top-fish value series do not overlap.",
            "source": ["count_catch_clean.csv", "catch_clean.csv", "luke_clean.csv"],
        }

    merged["difference"] = merged["left_value"] - merged["right_value"]

    crossover_year = None
    already_higher_from_start = False

    first_diff = merged.iloc[0]["difference"]
    if first_diff > 0:
        already_higher_from_start = True
        crossover_year = int(merged.iloc[0]["year"])
    else:
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
        "item_key": item_key,
        "crossover_year": crossover_year,
        "already_higher_from_start": already_higher_from_start,
        "data_points": merged.to_dict(orient="records"),
        "top_fish_reference": yearly_top_rows,
        "source": ["count_catch_clean.csv", "catch_clean.csv", "luke_clean.csv"],
    }