# Methodology and Assumptions — Pyhäjärvi Data Assistant

This document codifies the definitions, conventions, and analytical assumptions used by
the assistant. It is part of the knowledge base and is indexed for retrieval so that
definitional and "how is this calculated" questions can be answered with citations.

## Scope and data sources

The assistant works with three families of data about Lake Pyhäjärvi:

1. **Catch statistics** — fish catch per species in kilograms, roughly 2010–2025.
2. **Water-quality data** — chlorophyll-a, total phosphorus, total nitrogen, and
   temperature, measured at different depth zones (surface, mid, bottom).
3. **Commercial (Luke) statistics** — market quantity and price, used to estimate the
   economic value of catches.

Numeric tables live in cleaned CSV files and are queried through deterministic analytics
tools, not through this document. This document covers only definitions and methodology.

## Species and item conventions

- Species are identified by canonical English keys: perch, pike, bream, burbot, vendace,
  ruffe, whitefish, smelt, trout, roach, bleak, tench.
- Finnish names map to the same keys: ahven=perch, hauki=pike, lahna=bream, made=burbot,
  muikku=vendace, kiiski=ruffe, siika=whitefish, kuore=smelt, taimen=trout, särki=roach,
  salakka=bleak, suutari=tench.
- **"Crayfish" means the signal crayfish** (Finnish: *täplärapu*), tracked under the key
  `signal_crayfish`. It is a **count-based item**, not a weight-based fish.

## Weight-based vs count-based valuation

- Most fish are recorded by **weight (kg)**. Their economic value is estimated as
  `value = catch_kg × price_eur_per_kg`.
- **Signal crayfish is recorded by count (number of individuals), not weight.** Its value
  is therefore computed with **count-based valuation**: counted units converted to economic
  value using the appropriate per-unit price, rather than a per-kilogram price. Comparisons
  between crayfish and fish are made on the **estimated economic value (euros)** axis so the
  two unit systems are comparable.

## Price and missing-data handling

- Prices come from the commercial (Luke) statistics.
- When a price for a given species and year is **missing**, it is estimated by
  **interpolation** between known years, or by using the **nearest available year** when
  interpolation is not possible. Results report which source year and method were used.
- Missing values in trends and correlations are handled by aligning on the overlapping
  years that have data on both sides; the number of overlapping years used is reported.

## Water-quality depth zones

- Metrics are available at three depth zones: **surface**, **mid**, and **bottom**.
- Metric keys encode the zone and unit, e.g. `total_p_surface_ug_l` (surface total
  phosphorus, µg/L), `total_n_bottom_ug_l` (bottom total nitrogen, µg/L),
  `chlorophyll_a_surface_ug_l`, `temp_1m_c` (temperature at 1 m), `temp_bottom_c`.
- "Phosphorus depth zones" / "nitrogen depth zones" refer to the surface+mid+bottom group
  of that nutrient.

## Lag conventions for correlations

- Correlations may be computed with a year **lag**: "previous year" / "last year's" means
  lag = 1; "following year" / "next year" means lag = −1; otherwise lag = 0.

## Interpretation caveats

- Reported relationships are **correlations, not causation**.
- All figures are **annual**; sub-annual variation is not modeled.
- Economic values are **estimates** built from catch and price data, not audited revenue.
