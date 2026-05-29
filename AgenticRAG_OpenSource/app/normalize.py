"""Deterministic normalization of model-supplied arguments.

A local 14B model will not reliably emit canonical keys (e.g. "täplärapu" ->
"signal_crayfish", "chlorophyll" -> "chlorophyll_a_surface_ug_l"). Every
species/metric/item argument the agent passes is coerced through these maps
before it reaches the pandas-based analytics functions.

Extracted from app/chat.py so both the agent loop and app/agent_tools.py can
reuse them without a circular import.
"""
from __future__ import annotations

from typing import Any

from app.tools import tool_list_available_metrics


KNOWN_SPECIES_ALIASES: dict[str, str] = {
    "perch": "perch", "ahven": "perch",
    "pike": "pike", "hauki": "pike",
    "bream": "bream", "lahna": "bream",
    "burbot": "burbot", "made": "burbot",
    "vendace": "vendace", "muikku": "vendace",
    "ruffe": "ruffe", "kiiski": "ruffe",
    "whitefish": "whitefish", "siika": "whitefish",
    "smelt": "smelt", "kuore": "smelt",
    "trout": "trout", "taimen": "trout",
    "roach": "roach", "särki": "roach", "sarki": "roach",
    "bleak": "bleak", "salakka": "bleak",
    "tench": "tench", "suutari": "tench",
    "signal crayfish": "signal_crayfish",
    "signal_crayfish": "signal_crayfish",
    "crayfish": "signal_crayfish",
    "täplärapu": "signal_crayfish", "taplarapu": "signal_crayfish",
}

KNOWN_METRIC_ALIASES: dict[str, str] = {
    "chlorophyll": "chlorophyll_a_surface_ug_l",
    "chlorophyll-a": "chlorophyll_a_surface_ug_l",
    "chlorophyll a": "chlorophyll_a_surface_ug_l",
    "chlorophyll concentration": "chlorophyll_a_surface_ug_l",
    "temperature": "temp_1m_c",
    "surface temperature": "temp_1m_c",
    "bottom temperature": "temp_bottom_c",
    "phosphorus": "total_p_surface_ug_l",
    "total phosphorus": "total_p_surface_ug_l",
    "surface phosphorus": "total_p_surface_ug_l",
    "mid phosphorus": "total_p_mid_ug_l",
    "middle phosphorus": "total_p_mid_ug_l",
    "bottom phosphorus": "total_p_bottom_ug_l",
    "nitrogen": "total_n_surface_ug_l",
    "total nitrogen": "total_n_surface_ug_l",
    "surface nitrogen": "total_n_surface_ug_l",
    "mid nitrogen": "total_n_mid_ug_l",
    "middle nitrogen": "total_n_mid_ug_l",
    "bottom nitrogen": "total_n_bottom_ug_l",
}

METRIC_GROUP_ALIASES: dict[str, list[str]] = {
    "phosphorus depth zones": ["total_p_surface_ug_l", "total_p_mid_ug_l", "total_p_bottom_ug_l"],
    "phosphorus at different depth zones": ["total_p_surface_ug_l", "total_p_mid_ug_l", "total_p_bottom_ug_l"],
    "different depth zones phosphorus": ["total_p_surface_ug_l", "total_p_mid_ug_l", "total_p_bottom_ug_l"],
    "nitrogen depth zones": ["total_n_surface_ug_l", "total_n_mid_ug_l", "total_n_bottom_ug_l"],
}


def safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_species_key(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    # exact alias, else fall back to the canonical form if it's already a key
    return KNOWN_SPECIES_ALIASES.get(text, text.replace(" ", "_") if text.replace(" ", "_") in set(KNOWN_SPECIES_ALIASES.values()) else None)


def normalize_species_keys(values: Any) -> list[str]:
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
        return []
    out: list[str] = []
    for v in values:
        key = normalize_species_key(v)
        if key and key not in out:
            out.append(key)
    return out


def normalize_metric_key(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    available = tool_list_available_metrics()["metrics"]
    if text in available:
        return text
    return KNOWN_METRIC_ALIASES.get(text.lower())


def normalize_metric_keys(values: Any) -> list[str]:
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
        return []
    out: list[str] = []
    for v in values:
        key = normalize_metric_key(v)
        if key and key not in out:
            out.append(key)
    return out


def normalize_direction(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"increase", "increased", "growth", "grow", "up"}:
        return "increase"
    if text in {"decrease", "decreased", "decline", "drop", "down"}:
        return "decrease"
    return None
