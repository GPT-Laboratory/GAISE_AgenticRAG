from __future__ import annotations

from typing import Any

from analytics.series_engine import (
    align_two_series,
    build_combined_species_catch_series,
    build_metric_series,
    build_species_catch_series,
    compare_two_series,
    correlate_aligned_series,
    summarize_series,
)


def correlate_species_with_species(
    species_a: str,
    species_b: str,
    lag_years: int = 0,
) -> dict[str, Any]:
    left = build_species_catch_series(species_a)
    right = build_species_catch_series(species_b)

    aligned = align_two_series(left, right, right_lag_years=lag_years)
    corr = correlate_aligned_series(aligned)

    return {
        "species_a": species_a,
        "species_b": species_b,
        "lag_years": lag_years,
        **corr,
        "source_left": "catch_clean.csv",
        "source_right": "catch_clean.csv",
    }


def compare_species_trends(species_a: str, species_b: str) -> dict[str, Any]:
    left = build_species_catch_series(species_a)
    right = build_species_catch_series(species_b)

    comparison = compare_two_series(left, right)

    return {
        "species_a": species_a,
        "species_b": species_b,
        **comparison,
        "source": "catch_clean.csv",
    }


def correlate_species_with_metric(
    species_key: str,
    metric_key: str,
    lag_years: int = 0,
) -> dict[str, Any]:
    left = build_species_catch_series(species_key)
    right = build_metric_series(metric_key)

    aligned = align_two_series(left, right, right_lag_years=lag_years)
    corr = correlate_aligned_series(aligned)

    return {
        "species_key": species_key,
        "metric_key": metric_key,
        "lag_years": lag_years,
        **corr,
        "source_left": "catch_clean.csv",
        "source_right": "water_quality_clean.csv",
    }


def correlate_combined_species_with_metric(
    species_keys: list[str],
    metric_key: str,
    lag_years: int = 0,
) -> dict[str, Any]:
    left = build_combined_species_catch_series(species_keys)
    right = build_metric_series(metric_key)

    aligned = align_two_series(left, right, right_lag_years=lag_years)
    corr = correlate_aligned_series(aligned)

    return {
        "species_keys": species_keys,
        "metric_key": metric_key,
        "lag_years": lag_years,
        **corr,
        "source_left": "catch_clean.csv",
        "source_right": "water_quality_clean.csv",
    }


def correlate_combined_species_with_metric_group(
    species_keys: list[str],
    metric_keys: list[str],
    lag_years: int = 0,
) -> dict[str, Any]:
    results = []

    for metric_key in metric_keys:
        result = correlate_combined_species_with_metric(
            species_keys=species_keys,
            metric_key=metric_key,
            lag_years=lag_years,
        )
        results.append(result)

    valid = [r for r in results if r.get("correlation") is not None]
    valid = sorted(valid, key=lambda x: abs(x["correlation"]), reverse=True)

    return {
        "species_keys": species_keys,
        "metric_keys": metric_keys,
        "lag_years": lag_years,
        "results": results,
        "strongest_metric": valid[0]["metric_key"] if valid else None,
        "source_left": "catch_clean.csv",
        "source_right": "water_quality_clean.csv",
    }


def summarize_species_and_metric_trends(
    species_key: str,
    metric_key: str,
) -> dict[str, Any]:
    species_series = build_species_catch_series(species_key)
    metric_series = build_metric_series(metric_key)

    aligned = align_two_series(species_series, metric_series, right_lag_years=0)

    if aligned.empty:
        return {
            "species_key": species_key,
            "metric_key": metric_key,
            "message": "No overlapping years available for trend summary.",
            "source_left": "catch_clean.csv",
            "source_right": "water_quality_clean.csv",
        }

    species_summary = summarize_series(
        aligned[["year", "left_value"]].rename(columns={"left_value": "value"})
    )
    metric_summary = summarize_series(
        aligned[["year", "right_value"]].rename(columns={"right_value": "value"})
    )

    return {
        "species_key": species_key,
        "metric_key": metric_key,
        "species_summary": species_summary,
        "metric_summary": metric_summary,
        "aligned_data": aligned.to_dict(orient="records"),
        "source_left": "catch_clean.csv",
        "source_right": "water_quality_clean.csv",
    }