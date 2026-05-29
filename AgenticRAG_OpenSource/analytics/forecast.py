"""Lightweight predictive model: linear-trend forecast of species catch.

Deliberately simple and interpretable, as the challenge brief allows: an ordinary
least-squares linear regression of catch (kg) on year, projected a few years ahead.
This is a *trend extrapolation*, not a biological model — it assumes the historical
linear trend continues, which the returned `caveat` makes explicit.

Returns plain dicts (never raises for missing data) so the tool layer can turn the
result into a grounded, hedged natural-language answer.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression

from analytics.data_loader import load_catch_data

# Minimum historical points before a linear fit is meaningful.
_MIN_POINTS = 4
_MAX_HORIZON = 10


def forecast_species_catch(species_key: str, horizon: int = 3) -> dict:
    """Project a species' catch (kg) `horizon` years past the last observed year.

    Method: OLS linear regression of catch_kg on year. Negative projections are
    clamped to 0 (catch cannot be negative) and flagged in the caveat.
    """
    horizon = max(1, min(int(horizon or 3), _MAX_HORIZON))

    df = load_catch_data()
    sdf = (
        df[df["species_key"].str.lower() == str(species_key).lower()]
        .groupby("year", dropna=True)["catch_kg"]
        .sum()
        .reset_index()
        .sort_values("year")
    )

    if sdf.empty:
        return {"species_key": species_key, "message": f"No catch data found for species '{species_key}'."}
    if len(sdf) < _MIN_POINTS:
        return {
            "species_key": species_key,
            "message": (
                f"Only {len(sdf)} year(s) of data for '{species_key}' — too few to fit a "
                f"reliable trend (need at least {_MIN_POINTS})."
            ),
        }

    years = sdf["year"].to_numpy(dtype=float).reshape(-1, 1)
    catch = sdf["catch_kg"].to_numpy(dtype=float)

    model = LinearRegression().fit(years, catch)
    slope = float(model.coef_[0])          # kg per year
    r2 = float(model.score(years, catch))  # in-sample fit quality

    last_year = int(sdf["year"].max())
    future_years = np.arange(last_year + 1, last_year + 1 + horizon, dtype=float).reshape(-1, 1)
    preds = model.predict(future_years)

    clamped = False
    forecast = []
    for yr, val in zip(future_years.ravel(), preds):
        v = float(val)
        if v < 0:
            v = 0.0
            clamped = True
        forecast.append({"year": int(yr), "predicted_catch_kg": round(v, 1)})

    direction = "increasing" if slope > 0 else "decreasing" if slope < 0 else "flat"
    caveat = (
        "Linear trend extrapolation: assumes the historical year-over-year trend continues. "
        "Not a biological/ecological model; treat as indicative only."
    )
    if clamped:
        caveat += " Some projections fell below zero and were clamped to 0 kg."
    if r2 < 0.3:
        caveat += f" The linear fit is weak (R²={r2:.2f}), so the projection is low-confidence."

    return {
        "species_key": species_key,
        "model": "linear_regression",
        "fit_years": [int(sdf["year"].min()), last_year],
        "n_observations": int(len(sdf)),
        "slope_kg_per_year": round(slope, 1),
        "r_squared": round(r2, 3),
        "trend": direction,
        "history": [
            {"year": int(y), "catch_kg": round(float(c), 1)}
            for y, c in zip(sdf["year"], sdf["catch_kg"])
        ],
        "forecast": forecast,
        "caveat": caveat,
        "source": "catch_clean.csv",
    }
