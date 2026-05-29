from __future__ import annotations

from analytics.count_valuations import (
    compare_count_item_to_top_fish_by_year,
    compare_count_item_to_top_fish_over_time,
    find_count_item_vs_top_fish_crossover,
    summarize_count_item_value_trend,
)
from analytics.economic_rankings import (
    compare_species_values,
    get_top_value_species_by_year,
    get_top_value_species_over_time,
    get_value_crossover_year,
    has_top_value_species_changed_over_time,
    summarize_species_value_trend,
)
from analytics.series_engine import (
    build_total_catch_series,
    get_available_count_items,
    get_available_metrics,
    get_available_species,
    summarize_series,
)
from analytics.species_relationships import (
    compare_species_trends,
    correlate_combined_species_with_metric,
    correlate_combined_species_with_metric_group,
    correlate_species_with_metric,
    correlate_species_with_species,
    summarize_species_and_metric_trends,
)
from analytics.trends import (
    get_largest_species_change,
    get_species_trend,
    get_top_species_by_year,
    get_total_catch_for_year,
)
from analytics.valuations import estimate_species_value


def tool_list_available_species() -> dict:
    species = get_available_species()
    return {
        "species": species,
        "count": len(species),
        "source": "catch_clean.csv",
    }


def tool_list_available_count_items() -> dict:
    items = get_available_count_items()
    return {
        "items": items,
        "count": len(items),
        "source": "count_catch_clean.csv",
    }


def tool_list_available_metrics() -> dict:
    metrics = get_available_metrics()
    return {
        "metrics": metrics,
        "count": len(metrics),
        "source": "water_quality_clean.csv",
    }


def tool_get_total_catch_for_year(year: int) -> dict:
    return get_total_catch_for_year(year)


def tool_get_total_catch_trend() -> dict:
    series = build_total_catch_series()
    summary = summarize_series(series)
    return {
        **summary,
        "source": "catch_clean.csv",
    }


def tool_get_species_trend(species_key: str) -> dict:
    return get_species_trend(species_key)


def tool_get_top_species_by_year(year: int, limit: int = 5) -> dict:
    return get_top_species_by_year(year=year, limit=limit)


def tool_get_largest_species_change(direction: str = "increase") -> dict:
    return get_largest_species_change(direction=direction)


def tool_estimate_species_value(species_key: str, year: int) -> dict:
    return estimate_species_value(species_key=species_key, year=year)


def tool_compare_species_trends(species_a: str, species_b: str) -> dict:
    return compare_species_trends(species_a, species_b)


def tool_correlate_species_with_species(species_a: str, species_b: str, lag_years: int = 0) -> dict:
    return correlate_species_with_species(species_a, species_b, lag_years=lag_years)


def tool_correlate_species_with_metric(species_key: str, metric_key: str, lag_years: int = 0) -> dict:
    return correlate_species_with_metric(species_key, metric_key, lag_years=lag_years)


def tool_correlate_combined_species_with_metric(
    species_keys: list[str],
    metric_key: str,
    lag_years: int = 0,
) -> dict:
    return correlate_combined_species_with_metric(species_keys, metric_key, lag_years=lag_years)


def tool_correlate_combined_species_with_metric_group(
    species_keys: list[str],
    metric_keys: list[str],
    lag_years: int = 0,
) -> dict:
    return correlate_combined_species_with_metric_group(species_keys, metric_keys, lag_years=lag_years)


def tool_summarize_species_and_metric_trends(species_key: str, metric_key: str) -> dict:
    return summarize_species_and_metric_trends(species_key, metric_key)


def tool_get_top_value_species_by_year(year: int, limit: int = 5) -> dict:
    return get_top_value_species_by_year(year, limit=limit)


def tool_get_top_value_species_over_time() -> dict:
    return get_top_value_species_over_time()


def tool_has_top_value_species_changed_over_time() -> dict:
    return has_top_value_species_changed_over_time()


def tool_compare_species_values(species_a: str, species_b: str, year: int) -> dict:
    return compare_species_values(species_a, species_b, year)


def tool_get_value_crossover_year(species_a: str, species_b: str) -> dict:
    return get_value_crossover_year(species_a, species_b)


def tool_summarize_species_value_trend(species_key: str) -> dict:
    return summarize_species_value_trend(species_key)


def tool_summarize_count_item_value_trend(item_key: str) -> dict:
    return summarize_count_item_value_trend(item_key)


def tool_compare_count_item_to_top_fish_by_year(item_key: str, year: int) -> dict:
    return compare_count_item_to_top_fish_by_year(item_key, year)


def tool_compare_count_item_to_top_fish_over_time(item_key: str) -> dict:
    return compare_count_item_to_top_fish_over_time(item_key)


def tool_find_count_item_vs_top_fish_crossover(item_key: str) -> dict:
    return find_count_item_vs_top_fish_crossover(item_key)