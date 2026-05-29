"""The 11 consolidated agent tools exposed to the local LLM.

Each tool is a thin orchestration layer that (1) normalizes model-supplied args
via app/normalize.py, (2) calls the EXISTING analytics wrappers in app/tools.py
(no analytics logic is rewritten), and (3) returns {"result", "chart"} where
`chart` is a deterministic chart_data payload (or None) built in app/charts.py.

The 24 fine-grained wrappers are deliberately collapsed into 11 higher-level
tools with enum/list params, because a local 14B degrades sharply in
tool-selection accuracy past ~10-12 near-synonym tools.

Exports:
  TOOL_SCHEMAS  - OpenAI-format tool JSON, bound to the model via LangGraph
  DISPATCH      - {tool_name: callable(**kwargs) -> {"result", "chart"}}
"""
from __future__ import annotations

from typing import Any, Callable

from app import charts
from app.normalize import (
    normalize_direction,
    normalize_metric_keys,
    normalize_species_key,
    normalize_species_keys,
    safe_int,
)
from app.tools import (
    tool_compare_count_item_to_top_fish_by_year,
    tool_compare_count_item_to_top_fish_over_time,
    tool_compare_species_trends,
    tool_compare_species_values,
    tool_correlate_combined_species_with_metric,
    tool_correlate_combined_species_with_metric_group,
    tool_correlate_species_with_metric,
    tool_correlate_species_with_species,
    tool_estimate_species_value,
    tool_find_count_item_vs_top_fish_crossover,
    tool_get_largest_species_change,
    tool_get_species_trend,
    tool_get_top_species_by_year,
    tool_get_top_value_species_by_year,
    tool_get_top_value_species_over_time,
    tool_get_total_catch_for_year,
    tool_get_total_catch_trend,
    tool_get_value_crossover_year,
    tool_has_top_value_species_changed_over_time,
    tool_list_available_count_items,
    tool_list_available_metrics,
    tool_list_available_species,
    tool_summarize_count_item_value_trend,
    tool_summarize_species_and_metric_trends,
    tool_summarize_species_value_trend,
)
from analytics.document_search import search_documents
from analytics.forecast import forecast_species_catch


def _r(result: dict[str, Any], chart: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"result": result, "chart": chart}


def _missing(param: str) -> dict[str, Any]:
    return _r({"message": f"Missing required parameter: {param}."})


def _has_data(result: dict[str, Any]) -> bool:
    return isinstance(result, dict) and "message" not in result and "error" not in result


# --------------------------------------------------------------------------- #
# 1. list_data_dimensions
# --------------------------------------------------------------------------- #
def list_data_dimensions(kind: str | None = None, **_: Any) -> dict[str, Any]:
    kind = (kind or "").strip().lower()
    if kind == "species":
        return _r(tool_list_available_species())
    if kind == "count_items":
        return _r(tool_list_available_count_items())
    if kind == "metrics":
        return _r(tool_list_available_metrics())
    return _missing("kind (species|count_items|metrics)")


# --------------------------------------------------------------------------- #
# 2. get_catch
# --------------------------------------------------------------------------- #
def get_catch(
    mode: str | None = None,
    year: Any = None,
    species_key: Any = None,
    limit: Any = 5,
    **_: Any,
) -> dict[str, Any]:
    mode = (mode or "").strip().lower()
    year = safe_int(year)
    limit = safe_int(limit) or 5
    species_key = normalize_species_key(species_key)

    if mode == "total_for_year":
        if year is None:
            return _missing("year")
        return _r(tool_get_total_catch_for_year(year))

    if mode == "total_trend":
        result = tool_get_total_catch_trend()
        chart = charts.line_chart("Total catch over time", result["series"]) if _has_data(result) and result.get("series") else None
        return _r(result, chart)

    if mode == "top_by_year":
        if year is None:
            return _missing("year")
        result = tool_get_top_species_by_year(year=year, limit=limit)
        chart = charts.bar_chart(f"Top catch species in {year}", result["results"], "species_key", "catch_kg") if result.get("results") else None
        return _r(result, chart)

    if mode == "top_overall":
        from analytics.data_loader import load_catch_data

        df = load_catch_data()
        ranked = (
            df.groupby("species_key", dropna=True)["catch_kg"]
            .sum()
            .reset_index()
            .sort_values("catch_kg", ascending=False)
            .head(limit)
            .reset_index(drop=True)
        )
        result = {"results": ranked.to_dict(orient="records"), "source": "catch_clean.csv"}
        chart = charts.bar_chart("Top catch species over full period", result["results"], "species_key", "catch_kg")
        return _r(result, chart)

    if mode == "species_trend":
        if species_key is None:
            return _missing("species_key")
        result = tool_get_species_trend(species_key)
        chart = None
        if _has_data(result) and result.get("series"):
            chart = charts.line_chart(f"{species_key} catch over time", result["series"], x_key="year", y_key="catch_kg")
        return _r(result, chart)

    return _missing("mode (total_for_year|total_trend|top_by_year|top_overall|species_trend)")


# --------------------------------------------------------------------------- #
# 3. get_largest_change
# --------------------------------------------------------------------------- #
def get_largest_change(direction: Any = None, **_: Any) -> dict[str, Any]:
    direction = normalize_direction(direction)
    if direction is None:
        return _missing("direction (increase|decrease)")
    return _r(tool_get_largest_species_change(direction=direction))


# --------------------------------------------------------------------------- #
# 4. estimate_value
# --------------------------------------------------------------------------- #
def estimate_value(species_key: Any = None, year: Any = None, **_: Any) -> dict[str, Any]:
    species_key = normalize_species_key(species_key)
    year = safe_int(year)
    if species_key is None:
        return _missing("species_key")
    if year is None:
        return _missing("year")
    return _r(tool_estimate_species_value(species_key=species_key, year=year))


# --------------------------------------------------------------------------- #
# 5. rank_value_species
# --------------------------------------------------------------------------- #
def rank_value_species(year: Any = None, limit: Any = 5, check_changed: Any = False, **_: Any) -> dict[str, Any]:
    year = safe_int(year)
    limit = safe_int(limit) or 5

    if check_changed in (True, "true", "True", 1, "1"):
        result = tool_has_top_value_species_changed_over_time()
        chart = (
            charts.bar_chart("Yearly top-value species", result["yearly_leaders"], "year", "estimated_value_eur")
            if result.get("yearly_leaders")
            else None
        )
        return _r(result, chart)

    if year is not None:
        result = tool_get_top_value_species_by_year(year, limit=limit)
        chart = charts.bar_chart(f"Top value species in {year}", result["results"], "species_key", "estimated_value_eur") if result.get("results") else None
        return _r(result, chart)

    result = tool_get_top_value_species_over_time()
    if result.get("results"):
        top = result["results"][:limit]
        chart = charts.bar_chart("Top value species over full period", top, "species_key", "estimated_value_eur")
    else:
        chart = None
    return _r(result, chart)


# --------------------------------------------------------------------------- #
# 6. value_trend_or_compare
# --------------------------------------------------------------------------- #
def value_trend_or_compare(species_keys: Any = None, mode: str | None = None, year: Any = None, **_: Any) -> dict[str, Any]:
    mode = (mode or "").strip().lower()
    keys = normalize_species_keys(species_keys)
    year = safe_int(year)

    if mode == "trend":
        if not keys:
            return _missing("species_keys (one species)")
        result = tool_summarize_species_value_trend(keys[0])
        chart = charts.line_chart(f"{keys[0]} estimated value over time", result["series"]) if _has_data(result) and result.get("series") else None
        return _r(result, chart)

    if mode == "compare_year":
        if len(keys) < 2:
            return _missing("species_keys (two species)")
        if year is None:
            return _missing("year")
        return _r(tool_compare_species_values(keys[0], keys[1], year))

    if mode == "crossover":
        if len(keys) < 2:
            return _missing("species_keys (two species)")
        result = tool_get_value_crossover_year(keys[0], keys[1])
        chart = (
            charts.dual_line_chart(f"Estimated value crossover: {keys[0]} vs {keys[1]}", result["data_points"])
            if result.get("data_points")
            else None
        )
        return _r(result, chart)

    return _missing("mode (trend|compare_year|crossover)")


# --------------------------------------------------------------------------- #
# 7. compare_species_catch
# --------------------------------------------------------------------------- #
def compare_species_catch(species_keys: Any = None, mode: str | None = None, lag_years: Any = 0, **_: Any) -> dict[str, Any]:
    mode = (mode or "").strip().lower()
    keys = normalize_species_keys(species_keys)
    lag_years = safe_int(lag_years) or 0
    if len(keys) < 2:
        return _missing("species_keys (two species)")

    if mode == "trend_comparison":
        result = tool_compare_species_trends(keys[0], keys[1])
        chart = None
        if _has_data(result):
            chart = charts.multi_line_chart(
                f"{keys[0]} vs {keys[1]} catch trends",
                {keys[0]: result["series_a"]["series"], keys[1]: result["series_b"]["series"]},
            )
        return _r(result, chart)

    if mode == "correlation":
        result = tool_correlate_species_with_species(keys[0], keys[1], lag_years=lag_years)
        chart = charts.scatter_chart(f"{keys[0]} vs {keys[1]} catches", result["data_points"]) if _has_data(result) and result.get("data_points") else None
        return _r(result, chart)

    return _missing("mode (trend_comparison|correlation)")


# --------------------------------------------------------------------------- #
# 8. correlate_with_metric
# --------------------------------------------------------------------------- #
def correlate_with_metric(
    species_keys: Any = None,
    metric_keys: Any = None,
    lag_years: Any = 0,
    summary: Any = False,
    **kw: Any,
) -> dict[str, Any]:
    # The 14B model occasionally sends singular arg names; accept them as aliases
    # so a near-miss call still succeeds instead of erroring (which invites hallucination).
    species_keys = species_keys if species_keys is not None else kw.get("species") or kw.get("species_key")
    metric_keys = metric_keys if metric_keys is not None else kw.get("metric") or kw.get("metric_key") or kw.get("metric_keys")
    species = normalize_species_keys(species_keys)
    metrics = normalize_metric_keys(metric_keys)
    lag_years = safe_int(lag_years) or 0
    summary = summary in (True, "true", "True", 1, "1")

    if not species:
        return _missing("species_keys")
    if not metrics:
        return _missing("metric_keys")

    # 1 species + 1 metric, trend summary requested
    if summary and len(species) == 1 and len(metrics) == 1:
        result = tool_summarize_species_and_metric_trends(species[0], metrics[0])
        chart = None
        if _has_data(result) and result.get("aligned_data"):
            chart = charts.dual_line_chart(
                f"{species[0]} and {metrics[0]} over time", result["aligned_data"]
            )
        return _r(result, chart)

    # >=2 species + >=2 metrics -> metric group
    if len(species) >= 2 and len(metrics) >= 2:
        result = tool_correlate_combined_species_with_metric_group(species, metrics, lag_years=lag_years)
        chart = charts.grouped_results_chart(f"{' + '.join(species)} vs metric group correlations", result["results"]) if result.get("results") else None
        return _r(result, chart)

    # >=2 species + 1 metric -> combined
    if len(species) >= 2 and len(metrics) == 1:
        result = tool_correlate_combined_species_with_metric(species, metrics[0], lag_years=lag_years)
        chart = charts.scatter_chart(f"{' + '.join(species)} vs {metrics[0]} (lag {lag_years})", result["data_points"]) if _has_data(result) and result.get("data_points") else None
        return _r(result, chart)

    # 1 species + 1 metric -> single correlation
    result = tool_correlate_species_with_metric(species[0], metrics[0], lag_years=lag_years)
    chart = charts.scatter_chart(f"{species[0]} vs {metrics[0]} (lag {lag_years})", result["data_points"]) if _has_data(result) and result.get("data_points") else None
    return _r(result, chart)


# --------------------------------------------------------------------------- #
# 9. count_item_analysis
# --------------------------------------------------------------------------- #
def count_item_analysis(item_key: Any = None, mode: str | None = None, year: Any = None, **_: Any) -> dict[str, Any]:
    mode = (mode or "").strip().lower()
    item_key = normalize_species_key(item_key)
    year = safe_int(year)
    if item_key is None:
        return _missing("item_key")

    if mode == "value_trend":
        result = tool_summarize_count_item_value_trend(item_key)
        chart = charts.line_chart(f"{item_key} estimated value over time", result["series"]) if _has_data(result) and result.get("series") else None
        return _r(result, chart)

    if mode == "vs_top_fish_year":
        if year is None:
            return _missing("year")
        return _r(tool_compare_count_item_to_top_fish_by_year(item_key, year))

    if mode == "vs_top_fish_over_time":
        result = tool_compare_count_item_to_top_fish_over_time(item_key)
        chart = None
        if _has_data(result) and result.get("data_points"):
            chart = charts.dual_line_chart(
                f"{item_key} vs top fish value over time",
                result["data_points"],
                left_y_key="value",
                right_y_key="top_fish_value_eur",
            )
        return _r(result, chart)

    if mode == "vs_top_fish_crossover":
        result = tool_find_count_item_vs_top_fish_crossover(item_key)
        chart = (
            charts.dual_line_chart(f"{item_key} vs top fish species value over time", result["data_points"])
            if result.get("data_points")
            else None
        )
        return _r(result, chart)

    return _missing("mode (value_trend|vs_top_fish_year|vs_top_fish_over_time|vs_top_fish_crossover)")


# --------------------------------------------------------------------------- #
# 10. document_search
# --------------------------------------------------------------------------- #
def document_search(query: Any = None, **_: Any) -> dict[str, Any]:
    query = (str(query).strip() if query is not None else "")
    if not query:
        return _missing("query")
    return _r(search_documents(query))


# --------------------------------------------------------------------------- #
# 11. forecast_catch
# --------------------------------------------------------------------------- #
def forecast_catch(species_key: Any = None, horizon: Any = 3, **_: Any) -> dict[str, Any]:
    species_key = normalize_species_key(species_key)
    horizon = safe_int(horizon) or 3
    if species_key is None:
        return _missing("species_key")

    result = forecast_species_catch(species_key=species_key, horizon=horizon)
    chart = None
    if _has_data(result) and result.get("history") and result.get("forecast"):
        # Connect the lines: the forecast series starts from the last historical point.
        last_hist = result["history"][-1]
        forecast_line = [{"year": last_hist["year"], "catch_kg": last_hist["catch_kg"]}]
        forecast_line += [
            {"year": p["year"], "catch_kg": p["predicted_catch_kg"]} for p in result["forecast"]
        ]
        chart = charts.multi_line_chart(
            f"{species_key} catch forecast (linear trend)",
            {"historical": result["history"], "forecast": forecast_line},
        )
    return _r(result, chart)


# --------------------------------------------------------------------------- #
# Dispatch + schemas
# --------------------------------------------------------------------------- #
DISPATCH: dict[str, Callable[..., dict[str, Any]]] = {
    "list_data_dimensions": list_data_dimensions,
    "get_catch": get_catch,
    "get_largest_change": get_largest_change,
    "estimate_value": estimate_value,
    "rank_value_species": rank_value_species,
    "value_trend_or_compare": value_trend_or_compare,
    "compare_species_catch": compare_species_catch,
    "correlate_with_metric": correlate_with_metric,
    "count_item_analysis": count_item_analysis,
    "forecast_catch": forecast_catch,
    "document_search": document_search,
}


def _tool(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


_SPECIES_HINT = "Species name in English or Finnish (e.g. perch/ahven, vendace/muikku, signal crayfish/täplärapu)."
_METRIC_HINT = "Water-quality metric (e.g. chlorophyll, total phosphorus, surface nitrogen, bottom temperature)."

TOOL_SCHEMAS: list[dict[str, Any]] = [
    _tool(
        "list_data_dimensions",
        "List what data is available: fish species, count-based items (e.g. signal crayfish), or water-quality metrics.",
        {"kind": {"type": "string", "enum": ["species", "count_items", "metrics"]}},
        ["kind"],
    ),
    _tool(
        "get_catch",
        "Fish CATCH amounts in kilograms (not money). Total per year, total trend over time, top species in a year, top species overall, or one species' catch trend.",
        {
            "mode": {"type": "string", "enum": ["total_for_year", "total_trend", "top_by_year", "top_overall", "species_trend"]},
            "year": {"type": "integer", "description": "Required for total_for_year and top_by_year."},
            "species_key": {"type": "string", "description": f"Required for species_trend. {_SPECIES_HINT}"},
            "limit": {"type": "integer", "description": "How many species to rank (default 5)."},
        },
        ["mode"],
    ),
    _tool(
        "get_largest_change",
        "Which fish species had the largest relative increase or decrease in catch over the full period.",
        {"direction": {"type": "string", "enum": ["increase", "decrease"]}},
        ["direction"],
    ),
    _tool(
        "estimate_value",
        "Estimated economic VALUE in euros of one species in a given year (catch x price).",
        {
            "species_key": {"type": "string", "description": _SPECIES_HINT},
            "year": {"type": "integer"},
        },
        ["species_key", "year"],
    ),
    _tool(
        "rank_value_species",
        "Rank species by estimated economic VALUE (euros). Give a year for that year, omit year for the full period, or set check_changed=true to ask whether the most valuable species changed over time.",
        {
            "year": {"type": "integer"},
            "limit": {"type": "integer", "description": "How many to rank (default 5)."},
            "check_changed": {"type": "boolean", "description": "True to check if the top-value species changed over time."},
        },
        [],
    ),
    _tool(
        "value_trend_or_compare",
        "Estimated VALUE (euros) over time for species: one species' value trend, compare two species' value in a year, or find the year two species' values crossed over.",
        {
            "species_keys": {"type": "array", "items": {"type": "string"}, "description": f"One species for trend; two for compare_year/crossover. {_SPECIES_HINT}"},
            "mode": {"type": "string", "enum": ["trend", "compare_year", "crossover"]},
            "year": {"type": "integer", "description": "Required for compare_year."},
        },
        ["species_keys", "mode"],
    ),
    _tool(
        "compare_species_catch",
        "Compare two species' CATCH (kg): either compare their trends, or compute the correlation between their catches (optionally lagged).",
        {
            "species_keys": {"type": "array", "items": {"type": "string"}, "description": f"Exactly two species. {_SPECIES_HINT}"},
            "mode": {"type": "string", "enum": ["trend_comparison", "correlation"]},
            "lag_years": {"type": "integer", "description": "1 = previous year, -1 = next year, 0 = none (default)."},
        },
        ["species_keys", "mode"],
    ),
    _tool(
        "correlate_with_metric",
        "Correlate fish CATCH with water-quality metric(s). One species+one metric (set summary=true for a trend summary instead of correlation), several species combined vs one metric, or several species vs a group of metrics.",
        {
            "species_keys": {"type": "array", "items": {"type": "string"}, "description": _SPECIES_HINT},
            "metric_keys": {"type": "array", "items": {"type": "string"}, "description": _METRIC_HINT},
            "lag_years": {"type": "integer", "description": "1 = previous year, -1 = next year, 0 = none (default)."},
            "summary": {"type": "boolean", "description": "True for a side-by-side trend summary (one species + one metric)."},
        },
        ["species_keys", "metric_keys"],
    ),
    _tool(
        "count_item_analysis",
        "Analyze count-based items (e.g. signal crayfish, counted not weighed): value trend, compare to the top fish species in a year, over time, or find the crossover year when it became more valuable than the top fish.",
        {
            "item_key": {"type": "string", "description": "Count-based item, e.g. signal crayfish / täplärapu."},
            "mode": {"type": "string", "enum": ["value_trend", "vs_top_fish_year", "vs_top_fish_over_time", "vs_top_fish_crossover"]},
            "year": {"type": "integer", "description": "Required for vs_top_fish_year."},
        },
        ["item_key", "mode"],
    ),
    _tool(
        "forecast_catch",
        "PREDICT / forecast a fish species' future CATCH (kg) for the next few years using a linear-trend model. Use for questions about what the catch will be, future projections, or expected catch in upcoming years.",
        {
            "species_key": {"type": "string", "description": _SPECIES_HINT},
            "horizon": {"type": "integer", "description": "How many years ahead to project (default 3, max 10)."},
        },
        ["species_key"],
    ),
    _tool(
        "document_search",
        "Search the knowledge-base documents (reports, methodology, FAQs) for DEFINITIONS, assumptions, methodology, units, or any qualitative/explanatory question. Use this whenever the answer is not a number from the datasets.",
        {"query": {"type": "string", "description": "A focused natural-language search query."}},
        ["query"],
    ),
]
