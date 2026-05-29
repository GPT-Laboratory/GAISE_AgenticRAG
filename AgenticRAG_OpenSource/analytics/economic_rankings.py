from __future__ import annotations

from typing import Any

from analytics.series_engine import (
    build_all_species_value_table,
    build_species_value_series,
    find_crossover_year,
    summarize_series,
)


def get_top_value_species_by_year(year: int, limit: int = 5) -> dict[str, Any]:
    df = build_all_species_value_table(year=year).copy()

    if df.empty:
        return {
            "year": year,
            "results": [],
            "message": f"No estimated value data found for year {year}.",
            "source": ["catch_clean.csv", "luke_clean.csv"],
        }

    ranked = (
        df.groupby("species_key", dropna=True)["estimated_value_eur"]
        .sum(min_count=1)
        .reset_index()
        .sort_values("estimated_value_eur", ascending=False)
        .head(limit)
        .reset_index(drop=True)
    )

    return {
        "year": year,
        "results": ranked.to_dict(orient="records"),
        "source": ["catch_clean.csv", "luke_clean.csv"],
    }


def get_top_value_species_over_time() -> dict[str, Any]:
    df = build_all_species_value_table().copy()

    if df.empty:
        return {
            "results": [],
            "message": "No estimated value data found.",
            "source": ["catch_clean.csv", "luke_clean.csv"],
        }

    ranked = (
        df.groupby("species_key", dropna=True)["estimated_value_eur"]
        .sum(min_count=1)
        .reset_index()
        .sort_values("estimated_value_eur", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "results": ranked.to_dict(orient="records"),
        "source": ["catch_clean.csv", "luke_clean.csv"],
    }


def get_yearly_top_value_species() -> dict[str, Any]:
    df = build_all_species_value_table().copy()

    if df.empty:
        return {
            "results": [],
            "message": "No estimated value data found.",
            "source": ["catch_clean.csv", "luke_clean.csv"],
        }

    idx = df.groupby("year")["estimated_value_eur"].idxmax()
    top_each_year = df.loc[idx].sort_values("year").reset_index(drop=True)

    return {
        "results": top_each_year[["year", "species_key", "estimated_value_eur"]].to_dict(orient="records"),
        "source": ["catch_clean.csv", "luke_clean.csv"],
    }


def has_top_value_species_changed_over_time() -> dict[str, Any]:
    yearly = get_yearly_top_value_species()

    if "results" not in yearly or not yearly["results"]:
        return {
            "changed": False,
            "message": "No yearly top-value species data available.",
            "source": ["catch_clean.csv", "luke_clean.csv"],
        }

    species_sequence = [row["species_key"] for row in yearly["results"]]
    unique_species = sorted(set(species_sequence))

    return {
        "changed": len(unique_species) > 1,
        "unique_top_species": unique_species,
        "yearly_leaders": yearly["results"],
        "source": ["catch_clean.csv", "luke_clean.csv"],
    }


def compare_species_values(species_a: str, species_b: str, year: int) -> dict[str, Any]:
    df = build_all_species_value_table(year=year).copy()

    subset = df[df["species_key"].isin([species_a, species_b])].copy()

    if subset.empty:
        return {
            "year": year,
            "species_a": species_a,
            "species_b": species_b,
            "message": "No estimated value data found for the requested species in that year.",
            "source": ["catch_clean.csv", "luke_clean.csv"],
        }

    rows = (
        subset.groupby("species_key", dropna=True)["estimated_value_eur"]
        .sum(min_count=1)
        .reset_index()
    )

    value_map = {
        row["species_key"]: None if row["estimated_value_eur"] is None else float(row["estimated_value_eur"])
        for _, row in rows.iterrows()
    }

    winner = None
    a_val = value_map.get(species_a)
    b_val = value_map.get(species_b)

    if a_val is not None and b_val is not None:
        if a_val > b_val:
            winner = species_a
        elif b_val > a_val:
            winner = species_b
        else:
            winner = "tie"

    return {
        "year": year,
        "species_a": species_a,
        "species_b": species_b,
        "value_a_eur": a_val,
        "value_b_eur": b_val,
        "winner": winner,
        "source": ["catch_clean.csv", "luke_clean.csv"],
    }


def get_value_crossover_year(species_a: str, species_b: str) -> dict[str, Any]:
    series_a = build_species_value_series(species_a)
    series_b = build_species_value_series(species_b)

    crossover = find_crossover_year(series_a, series_b)

    return {
        "species_a": species_a,
        "species_b": species_b,
        "crossover_year": crossover.get("crossover_year"),
        "data_points": crossover.get("data_points", []),
        "source": ["catch_clean.csv", "luke_clean.csv"],
    }


def summarize_species_value_trend(species_key: str) -> dict[str, Any]:
    series = build_species_value_series(species_key)
    summary = summarize_series(series)

    return {
        "species_key": species_key,
        **summary,
        "source": ["catch_clean.csv", "luke_clean.csv"],
    }