from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"


def load_catch_data() -> pd.DataFrame:
    path = PROCESSED_DIR / "catch_clean.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing processed file: {path}")
    return pd.read_csv(path)


def load_luke_data() -> pd.DataFrame:
    path = PROCESSED_DIR / "luke_clean.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing processed file: {path}")
    return pd.read_csv(path)