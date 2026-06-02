"""
Custom tools for the Bio AI fishery assistant.

This module wraps the existing analytics functions located in the
``analytics`` folder from the candidate's submission into
``crewai.tools.BaseTool`` subclasses. Each tool exposes a descriptive
``name`` and ``description`` which makes it discoverable by an LLM and
implements a ``_run`` method that performs the deterministic
computation.

By encapsulating the analytics in tools you enable CrewAI agents to
invoke complex domain logic while maintaining clear separation
between the planning, reasoning and execution layers. Tools return
Python dictionaries and lists which CrewAI will automatically
serialise to JSON for the language model.

The functions imported below originate from the existing
``analytics`` package distributed with this exercise. We import them
via relative imports to ensure they are available when this module is
executed. If you add new analytics functions you can create
additional tool classes following the same pattern.
"""

from __future__ import annotations

from typing import Any

from crewai.tools import BaseTool

# Import the analytics functions from the provided package
from analytics.trends import (
    list_available_species,
    get_total_catch_by_year,
    get_total_catch_for_year,
    get_species_trend,
    get_top_species_by_year,
    get_largest_species_change,
)
from analytics.economic_rankings import (
    get_top_value_species_by_year,
    get_top_value_species_over_time,
    get_value_crossover_year,
    summarize_species_value_trend,
    compare_species_values,
)
from analytics.valuations import estimate_species_value
from analytics.relationships import (
    correlate_species_with_water_metric,
    correlate_combined_species_with_water_metric,
)


class ListSpeciesTool(BaseTool):
    """Tool to list all species available in the catch dataset."""
    name: str = "list_species"
    description: str = "List all fish and crayfish species in the catch dataset."

    def _run(self) -> Any:
        return list_available_species()


class TotalCatchTrendTool(BaseTool):
    """Tool to compute the total catch trend across all species."""
    name: str = "total_catch_trend"
    description: str = "Return a list of dictionaries with the total catch (kg) per year."

    def _run(self) -> Any:
        return get_total_catch_by_year()


class TotalCatchForYearTool(BaseTool):
    """Tool to compute the total catch for a specific year."""
    name: str = "total_catch_for_year"
    description: str = "Return total catch (kg) for a given year."

    def _run(self, year: int) -> Any:
        return get_total_catch_for_year(year)


class SpeciesTrendTool(BaseTool):
    """Tool to compute the catch trend for a specific species."""
    name: str = "species_trend"
    description: str = "Return catch trend information for a species, including start/end values, percent change and annual series."

    def _run(self, species_key: str) -> Any:
        return get_species_trend(species_key)


class TopSpeciesByYearTool(BaseTool):
    """Tool to list top species by catch for a given year."""
    name: str = "top_species_by_year"
    description: str = "Return the top N species ranked by catch in a given year."

    def _run(self, year: int, limit: int = 5) -> Any:
        return get_top_species_by_year(year, limit)


class LargestSpeciesChangeTool(BaseTool):
    """Tool to find the species with the largest increase or decrease over the dataset."""
    name: str = "largest_species_change"
    description: str = "Return the species with the largest percentage change in catch over the measurement period."

    def _run(self, direction: str = "increase") -> Any:
        return get_largest_species_change(direction)


class EstimateSpeciesValueTool(BaseTool):
    """Tool to estimate the economic value of a species for a specific year."""
    name: str = "estimate_species_value"
    description: str = "Return the estimated value (EUR) of a species in a given year based on catch and price data."

    def _run(self, species_key: str, year: int) -> Any:
        return estimate_species_value(species_key, year)


class TopValueSpeciesByYearTool(BaseTool):
    """Tool to list the most valuable species in a given year."""
    name: str = "top_value_species_by_year"
    description: str = "Return the top N species by estimated economic value in a given year."

    def _run(self, year: int, limit: int = 5) -> Any:
        return get_top_value_species_by_year(year, limit)


class TopValueSpeciesOverTimeTool(BaseTool):
    """Tool to list species ranked by total value over all years."""
    name: str = "top_value_species_over_time"
    description: str = "Return species ranked by the sum of their estimated values across the time series."

    def _run(self, limit: int = 5) -> Any:
        result = get_top_value_species_over_time()
        # Optionally limit the number of results if requested
        if limit is not None and "results" in result:
            result["results"] = result["results"][:limit]
        return result


class ValueCrossoverYearTool(BaseTool):
    """Tool to determine when one species became more valuable than another."""
    name: str = "value_crossover_year"
    description: str = "Return the first year where species A's estimated value exceeded species B's value."

    def _run(self, species_a: str, species_b: str) -> Any:
        return get_value_crossover_year(species_a, species_b)


class SpeciesValueTrendTool(BaseTool):
    """Tool to summarise the trend of a species' value over time."""
    name: str = "species_value_trend"
    description: str = "Return summary statistics for a species' estimated value trend over the time series."

    def _run(self, species_key: str) -> Any:
        return summarize_species_value_trend(species_key)


class CompareSpeciesValuesTool(BaseTool):
    """Tool to compare the estimated values of two species for a particular year."""
    name: str = "compare_species_values"
    description: str = "Compare the estimated values of two species in a given year, returning the winner and values."

    def _run(self, species_a: str, species_b: str, year: int) -> Any:
        return compare_species_values(species_a, species_b, year)


class SpeciesMetricCorrelationTool(BaseTool):
    """Tool to compute correlation between species catch and a water quality metric."""
    name: str = "species_metric_correlation"
    description: str = "Return correlation statistics between a species' catch and a given water quality metric, optionally with lag."

    def _run(self, species_key: str, metric_key: str, lag_years: int = 0) -> Any:
        return correlate_species_with_water_metric(species_key, metric_key, lag_years)


class CombinedSpeciesMetricCorrelationTool(BaseTool):
    """Tool to compute correlation between combined species catch and a water quality metric."""
    name: str = "combined_species_metric_correlation"
    description: str = "Return correlation between the combined catch of multiple species and a water quality metric, optionally with lag."

    def _run(self, species_keys: list[str], metric_key: str, lag_years: int = 0) -> Any:
        return correlate_combined_species_with_water_metric(species_keys, metric_key, lag_years)
