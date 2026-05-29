"""Chart payload builders shared by the agent tools.

These produce the `chart_data` shapes the React UI (ui/src/components/ChatPanel.jsx)
renders. The agent LLM only chooses *which tool* to run; chart construction stays
deterministic here so numbers are never invented by the model.
"""
from __future__ import annotations

from typing import Any


def line_chart(title: str, data: list[dict[str, Any]], x_key: str = "year", y_key: str = "value") -> dict[str, Any]:
    return {"chart_type": "line", "title": title, "x_key": x_key, "y_key": y_key, "data": data}


def bar_chart(title: str, data: list[dict[str, Any]], x_key: str, y_key: str) -> dict[str, Any]:
    return {"chart_type": "bar", "title": title, "x_key": x_key, "y_key": y_key, "data": data}


def scatter_chart(title: str, data: list[dict[str, Any]]) -> dict[str, Any]:
    return {"chart_type": "scatter", "title": title, "x_key": "right_value", "y_key": "left_value", "data": data}


def multi_line_chart(title: str, series: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    return {"chart_type": "multi_line", "title": title, "series": series}


def dual_line_chart(
    title: str,
    data: list[dict[str, Any]],
    x_key: str = "year",
    left_y_key: str = "left_value",
    right_y_key: str = "right_value",
) -> dict[str, Any]:
    return {
        "chart_type": "dual_line",
        "title": title,
        "data": data,
        "x_key": x_key,
        "left_y_key": left_y_key,
        "right_y_key": right_y_key,
    }


def grouped_results_chart(title: str, data: list[dict[str, Any]]) -> dict[str, Any]:
    return {"chart_type": "grouped_results", "title": title, "data": data}
