"""Load the realtor.com USA Real Estate CSV and build the NC homes table.

Usage:  python src/load_data.py   (prints a summary of each filtering step)
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "realtor-data.csv"
STATE = "North Carolina"

# Columns that together identify one listing (used to drop reposted duplicates)
DEDUP_KEYS = ["price", "bed", "bath", "house_size", "street", "zip_code", "city"]


def load_raw() -> pd.DataFrame:
    if not RAW_CSV.exists():
        raise FileNotFoundError(
            f"{RAW_CSV} not found. See README.md -> 'Getting the data'."
        )
    return pd.read_csv(RAW_CSV)


def load_nc_homes(df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (nc_homes, steps).

    nc_homes: NC homes that are for sale or sold, with price/bed/bath/size present
              (land lots dropped), duplicates removed.
    steps:    row count after each filtering step, for the report.
    """
    if df is None:
        df = load_raw()
    steps = [("All US listings", len(df))]

    nc = df[df["state"] == STATE]
    steps.append(("North Carolina only", len(nc)))

    nc = nc[nc["status"].isin(["for_sale", "sold"])]
    steps.append(("Status for_sale or sold (drop ready_to_build)", len(nc)))

    nc = nc.dropna(subset=["price", "bed", "bath", "house_size"])
    steps.append(("Has price, bed, bath, house_size (drop land lots)", len(nc)))

    nc = nc.drop_duplicates(subset=DEDUP_KEYS)
    steps.append(("Duplicates removed", len(nc)))

    nc = nc.copy()
    # ZIPs are stored as floats (28207.0); make them 5-digit strings, keep missing as NA
    nc["zip_code"] = nc["zip_code"].astype("Int64").astype("string").str.zfill(5)
    nc = nc.reset_index(drop=True)
    return nc, pd.DataFrame(steps, columns=["step", "rows"])


def main() -> None:
    nc, steps = load_nc_homes()
    print(steps.to_string(index=False), "\n")
    print("Status counts:")
    print(nc["status"].value_counts().to_string(), "\n")
    cols = ["price", "bed", "bath", "house_size", "acre_lot"]
    print(nc[cols].describe().round(1).to_string())


if __name__ == "__main__":
    main()
