from __future__ import annotations

from typing import Any

from .series_engine import build_combined_species_catch_series


def get_combined_species_catch(species_keys: list[str]) -> dict[str, Any]:
    """Return a combined catch time series for multiple species.

    The relationships module expects this helper when computing
    correlations between grouped species catch and water-quality metrics.
    """
    series = build_combined_species_catch_series(species_keys)

    if series.empty:
        return {
            "species_keys": species_keys,
            "series": [],
            "message": "No combined catch data found for the requested species.",
            "source": "catch_clean.csv",
        }

    return {
        "species_keys": species_keys,
        "series": series.rename(columns={"value": "catch_kg"}).to_dict(orient="records"),
        "source": "catch_clean.csv",
    }
