"""Load the realtor.com USA Real Estate CSV and print a first-look summary for NC.

Usage:  python src/load_data.py
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "realtor-data.csv"
STATE = "North Carolina"


def load_raw() -> pd.DataFrame:
    if not RAW_CSV.exists():
        raise FileNotFoundError(
            f"{RAW_CSV} not found. See README.md -> 'Getting the data'."
        )
    return pd.read_csv(RAW_CSV)


def main() -> None:
    df = load_raw()
    print(f"Full dataset: {df.shape[0]:,} rows x {df.shape[1]} columns\n")
    print("Columns and dtypes:")
    print(df.dtypes.to_string(), "\n")

    nc = df[df["state"] == STATE]
    print(f"{STATE} rows: {len(nc):,}\n")

    print("NC missing values per column:")
    print(nc.isna().sum().to_string(), "\n")

    print("NC status counts:")
    print(nc["status"].value_counts().to_string(), "\n")

    usable = nc.dropna(subset=["price", "bed", "bath", "house_size"])
    print(f"NC rows with price, bed, bath and house_size: {len(usable):,}\n")

    print("NC summary statistics:")
    cols = ["price", "bed", "bath", "house_size", "acre_lot"]
    print(nc[cols].describe().round(1).to_string(), "\n")

    print("Top 10 NC cities:")
    print(nc["city"].value_counts().head(10).to_string())


if __name__ == "__main__":
    main()
