from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"


SPECIES_MAP: Dict[str, Dict[str, str]] = {
    "AHVEN": {"english": "Perch", "key": "perch"},
    "HAUKI": {"english": "Pike", "key": "pike"},
    "LAHNA": {"english": "Bream", "key": "bream"},
    "MADE": {"english": "Burbot", "key": "burbot"},
    "MUIKKU": {"english": "Vendace", "key": "vendace"},
    "KIISKI": {"english": "Ruffe", "key": "ruffe"},
    "SIIKA": {"english": "Whitefish", "key": "whitefish"},
    "KUORE": {"english": "Smelt", "key": "smelt"},
    "TAIMEN": {"english": "Trout", "key": "trout"},
    "SÄRKI": {"english": "Roach", "key": "roach"},
    "SARKI": {"english": "Roach", "key": "roach"},
    "SALAKKA": {"english": "Bleak", "key": "bleak"},
    "SUUTARI": {"english": "Tench", "key": "tench"},
    "KUHA": {"english": "Pike-perch", "key": "pike_perch"},
    "LOHI": {"english": "Salmon", "key": "salmon"},
    "TÄPLÄRAPU": {"english": "Signal crayfish", "key": "signal_crayfish"},
    "TAPLARAPU": {"english": "Signal crayfish", "key": "signal_crayfish"},
    "JOKIRAPU": {"english": "Noble crayfish", "key": "noble_crayfish"},
    "MUUT KALAT": {"english": "Other fish", "key": "other_fish"},
}

def ensure_dirs() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def normalize_text(value) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text == "":
        return None
    return text


def clean_numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str)
        .str.replace("\xa0", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False)
        .replace({"nan": None, "None": None, "..": None, "—": None, "-": None}),
        errors="coerce",
    )


def find_header_row_by_value(df: pd.DataFrame, expected_value: str) -> int:
    for idx in range(len(df)):
        row_values = [normalize_text(v) for v in df.iloc[idx].tolist()]
        if expected_value in row_values:
            return idx
    raise ValueError(f"Could not find header row containing '{expected_value}'.")


def _read_catch_source_table() -> pd.DataFrame:
    path = RAW_DIR / "TAUlle BioAItyohon pyhajarven saalistilasto 2010_2025.xlsx"
    raw = pd.read_excel(path, header=None, sheet_name=0)

    header_row = find_header_row_by_value(raw, "KALALAJI")
    header_values = raw.iloc[header_row].tolist()

    columns = []
    for i, value in enumerate(header_values):
        if i == 1:
            columns.append("species")
        elif i >= 2 and normalize_text(value) is not None:
            year_value = clean_numeric_series(pd.Series([value])).iloc[0]
            columns.append(str(int(year_value)) if pd.notna(year_value) else f"col_{i}")
        else:
            columns.append(f"col_{i}")

    df = raw.iloc[header_row + 1 :].copy()
    df.columns = columns

    keep_cols = ["species"] + [c for c in df.columns if c.isdigit()]
    df = df[keep_cols].copy()

    df["species"] = df["species"].apply(normalize_text)
    df = df[df["species"].notna()].copy()
    df["species"] = df["species"].astype(str).str.strip().str.upper()

    return df


def build_catch_clean() -> pd.DataFrame:
    df = _read_catch_source_table()

    df_long = df.melt(
        id_vars=["species"],
        var_name="year",
        value_name="catch_kg",
    )

    df_long["year"] = pd.to_numeric(df_long["year"], errors="coerce").astype("Int64")
    df_long["catch_kg"] = clean_numeric_series(df_long["catch_kg"])

    excluded_rows = {"YHTEENSÄ", "RAPUSAALIS KPL", "SUMMA", "TOTAL"}
    df_long = df_long[~df_long["species"].isin(excluded_rows)].copy()

    df_long["species_english"] = df_long["species"].map(
        lambda x: SPECIES_MAP.get(x, {}).get("english")
    )
    df_long["species_key"] = df_long["species"].map(
        lambda x: SPECIES_MAP.get(x, {}).get("key")
    )

    unknown_species = sorted(df_long.loc[df_long["species_key"].isna(), "species"].dropna().unique())
    if unknown_species:
        raise ValueError(f"Unmapped species in catch data: {unknown_species}")

    df_long = df_long.dropna(subset=["year", "catch_kg"]).copy()
    df_long["year"] = df_long["year"].astype(int)

    df_long = df_long[
        ["year", "species", "species_english", "species_key", "catch_kg"]
    ].sort_values(["species_key", "year"]).reset_index(drop=True)

    df_long.to_csv(PROCESSED_DIR / "catch_clean.csv", index=False)
    return df_long


def build_count_catch_clean() -> pd.DataFrame:
    """
    Creates:
        data/processed/count_catch_clean.csv

    Source:
        Catch source file row RAPUSAALIS KPL

    Output:
        year,item_key,count_units,unit
    """
    df = _read_catch_source_table()

    count_rows = {"RAPUSAALIS KPL"}
    df = df[df["species"].isin(count_rows)].copy()

    if df.empty:
        raise ValueError("Could not find RAPUSAALIS KPL row in the catch source file.")

    df_long = df.melt(
        id_vars=["species"],
        var_name="year",
        value_name="count_units",
    )

    df_long["year"] = pd.to_numeric(df_long["year"], errors="coerce")
    df_long["count_units"] = clean_numeric_series(df_long["count_units"])

    df_long = df_long.dropna(subset=["year", "count_units"]).copy()
    df_long["year"] = df_long["year"].astype(int)

    df_long["item_key"] = "signal_crayfish"
    df_long["unit"] = "kpl"

    out = (
        df_long[["year", "item_key", "count_units", "unit"]]
        .sort_values("year")
        .reset_index(drop=True)
    )

    out.to_csv(PROCESSED_DIR / "count_catch_clean.csv", index=False)
    return out


def build_water_quality_clean() -> pd.DataFrame:
    path = RAW_DIR / "TAUlle BioAItyohon pyhaj syvanteen vedenlaatutietoa.xlsx"
    raw = pd.read_excel(path, header=None, sheet_name=0)

    header_row = find_header_row_by_value(raw, "vuosi")
    header = [normalize_text(v) for v in raw.iloc[header_row].tolist()]

    df = raw.iloc[header_row + 1 :].copy()
    df.columns = header

    rename_map = {
        "vuosi": "year",
        "Keskiarvo  / syv v 1 Klorofylli-a µg/l": "chlorophyll_a_surface_ug_l",
        "Keskiarvo  / syv v 1 Kokonaistyppi, suodattamaton µg/l": "total_n_surface_ug_l",
        "Keskiarvo  / syv v 1 Kokonaisfosfori, suodattamaton µg/l": "total_p_surface_ug_l",
        "Keskiarvo  / syv v 2 Kokonaistyppi, suodattamaton µg/l": "total_n_mid_ug_l",
        "Keskiarvo  / syv v 2 Kokonaisfosfori, suodattamaton µg/l": "total_p_mid_ug_l",
        "Keskiarvo  / syv v 3 Kokonaistyppi, suodattamaton µg/l": "total_n_bottom_ug_l",
        "Keskiarvo  / syv v 3 Kokonaisfosfori, suodattamaton µg/l": "total_p_bottom_ug_l",
        "Keskiarvo  / lämpötila 1m": "temp_1m_c",
        "Keskiarvo  / lämpötila pohja": "temp_bottom_c",
    }

    df = df.rename(columns=rename_map)

    expected_cols = list(rename_map.values())
    df = df[expected_cols].copy()

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"]).copy()
    df["year"] = df["year"].astype(int)

    for col in df.columns:
        if col != "year":
            df[col] = clean_numeric_series(df[col])

    df = df.sort_values("year").reset_index(drop=True)
    df.to_csv(PROCESSED_DIR / "water_quality_clean.csv", index=False)
    return df


def build_luke_clean() -> pd.DataFrame:
    path = RAW_DIR / "TAUlle BioAItyohon luke 0600_kausis_20260326-134254.xlsx"
    raw = pd.read_excel(path, header=None, sheet_name=0)

    species_row = 3
    species_header = [normalize_text(v) for v in raw.iloc[species_row].tolist()]

    species_columns = []
    for idx, value in enumerate(species_header):
        if idx == 0:
            species_columns.append("unit")
        elif idx == 1:
            species_columns.append("year")
        else:
            species_columns.append(value if value else f"col_{idx}")

    # 🔹 Helper to extract any block
    def parse_block(start_row: int, end_row: int, value_name: str) -> pd.DataFrame:
        block = raw.iloc[start_row:end_row].copy()
        block.columns = species_columns

        block = block.drop(columns=["unit"], errors="ignore")
        block["year"] = pd.to_numeric(block["year"], errors="coerce")
        block = block.dropna(subset=["year"]).copy()
        block["year"] = block["year"].astype(int)

        value_cols = [c for c in block.columns if c != "year"]
        block[value_cols] = block[value_cols].replace("..", pd.NA)

        long_df = block.melt(
            id_vars=["year"],
            var_name="species_finnish_raw",
            value_name=value_name,
        )

        long_df[value_name] = clean_numeric_series(long_df[value_name])
        return long_df

    # 🔹 Parse ALL THREE blocks
    qty_kg_df = parse_block(4, 49, "quantity_1000kg")
    value_df = parse_block(49, 94, "value_1000eur")
    qty_kpl_df = parse_block(94, 139, "quantity_1000kpl")  # 🔥 NEW

    # 🔹 Merge all
    luke = qty_kg_df.merge(value_df, on=["year", "species_finnish_raw"], how="outer")
    luke = luke.merge(qty_kpl_df, on=["year", "species_finnish_raw"], how="outer")

    # 🔹 Normalize species names
    species_rename = {
        "Muikku": "MUIKKU",
        "Siika": "SIIKA",
        "Kuha": "KUHA",
        "Ahven": "AHVEN",
        "Hauki": "HAUKI",
        "Made": "MADE",
        "Särki": "SÄRKI",
        "Sarki": "SÄRKI",
        "Lahna": "LAHNA",
        "Kuore": "KUORE",
        "Taimen": "TAIMEN",
        "Lohi": "LOHI",
        "Muut kalat": "MUUT KALAT",
        "Täplärapu": "TÄPLÄRAPU",
        "Taplarapu": "TÄPLÄRAPU",
        "Jokirapu": "JOKIRAPU",
    }

    luke["species_finnish"] = luke["species_finnish_raw"].replace(species_rename)
    luke["species_english"] = luke["species_finnish"].map(
        lambda x: SPECIES_MAP.get(x, {}).get("english")
    )
    luke["species_key"] = luke["species_finnish"].map(
        lambda x: SPECIES_MAP.get(x, {}).get("key")
    )

    # 🔹 Convert units
    luke["quantity_kg"] = luke["quantity_1000kg"] * 1000
    luke["value_eur"] = luke["value_1000eur"] * 1000
    luke["quantity_units"] = luke["quantity_1000kpl"] * 1000

    # 🔹 Compute prices
    luke["price_eur_per_kg"] = luke["value_eur"] / luke["quantity_kg"].replace(0, pd.NA)
    luke["price_eur_per_unit"] = luke["value_eur"] / luke["quantity_units"].replace(0, pd.NA)

    # 🔹 Final columns
    luke = luke[
        [
            "year",
            "species_finnish",
            "species_english",
            "species_key",
            "quantity_kg",
            "quantity_units",
            "value_eur",
            "price_eur_per_kg",
            "price_eur_per_unit",  # 🔥 NEW IMPORTANT COLUMN
        ]
    ].sort_values(["species_key", "year"]).reset_index(drop=True)

    luke.to_csv(PROCESSED_DIR / "luke_clean.csv", index=False)

    return luke


def main() -> None:
    ensure_dirs()

    build_catch_clean()
    build_count_catch_clean()
    build_water_quality_clean()
    build_luke_clean()

    print("Done. Generated files:")
    for name in [
        "catch_clean.csv",
        "count_catch_clean.csv",
        "water_quality_clean.csv",
        "luke_clean.csv",
    ]:
        print(f" - {PROCESSED_DIR / name}")


if __name__ == "__main__":
    main()