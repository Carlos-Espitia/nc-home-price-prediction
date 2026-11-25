"""Preprocessing pipeline: NC homes -> cleaned, split, feature-engineered data.

Usage:  python src/preprocess.py   (builds and saves data/processed/{train,val,test}.csv)

Steps (decided Nov 25, see NOTES.md):
1. Load NC homes (for_sale + sold, land lots and duplicates removed)  -> load_data.py
2. Remove impossible values (outliers)
3. Split 70% train / 15% validation / 15% test (fixed seed)
4. Fill missing lot size with the TRAINING median + add a "lot was missing" flag
5. Location feature: ZIP median log($/sqft), smoothed toward city/state values,
   computed out-of-fold for training rows and from the full training set otherwise
6. Log-transform price (target), house size and lot size
7. Standardize features using TRAINING mean/std only (Standardize class from HW)
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

from load_data import ROOT, load_nc_homes

SEED = 42
PROCESSED_DIR = ROOT / "data" / "processed"

# Keep rows inside these limits (inclusive). None = no limit on that side.
LIMITS = {
    "price": (10_000, 5_000_000),
    "house_size": (300, 15_000),
    "bed": (None, 10),
    "bath": (None, 10),
    "acre_lot": (None, 100),  # missing lot sizes are kept and filled later
}

SMOOTHING_M = 10  # a ZIP with n homes gets weight n / (n + 10) on its own median
N_FOLDS = 5

TARGET = "log_price"
FEATURES = [
    "log_house_size",
    "bed",
    "bath",
    "log_acre_lot",
    "acre_lot_missing",
    "sold",
    "zip_log_ppsf",
]
# Extra columns kept for error analysis (not model inputs)
META = ["price", "house_size", "acre_lot", "status", "city", "zip_code"]


# ---------------------------------------------------------------- outliers
def remove_outliers(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop rows outside LIMITS. Returns (clean_df, table of rows removed per rule)."""
    keep = pd.Series(True, index=df.index)
    report = []
    for col, (lo, hi) in LIMITS.items():
        bad = pd.Series(False, index=df.index)
        if lo is not None:
            bad |= df[col] < lo
        if hi is not None:
            bad |= df[col] > hi
        bad &= df[col].notna()
        rule = f"{col} outside [{lo if lo is not None else '-'}, {hi if hi is not None else '-'}]"
        report.append((rule, int(bad.sum()), int((bad & keep).sum())))
        keep &= ~bad
    table = pd.DataFrame(report, columns=["rule", "rows failing", "removed (not already removed)"])
    return df[keep].reset_index(drop=True), table


# ---------------------------------------------------------------- split
def split(df: pd.DataFrame, seed: int = SEED):
    """70% train, 15% validation, 15% test."""
    trn, rest = train_test_split(df, train_size=0.70, random_state=seed)
    val, tst = train_test_split(rest, train_size=0.50, random_state=seed)
    return trn.reset_index(drop=True), val.reset_index(drop=True), tst.reset_index(drop=True)


# ---------------------------------------------------------------- ZIP location feature
def _blend(stats: pd.DataFrame, prior, m: int = SMOOTHING_M) -> pd.Series:
    """(n * group_median + m * prior) / (n + m): small groups lean on the prior."""
    return (stats["size"] * stats["median"] + m * prior) / (stats["size"] + m)


def fit_zip_encoding(fit_df: pd.DataFrame, m: int = SMOOTHING_M) -> dict:
    """Learn smoothed median log($/sqft) per ZIP (falls back to city, then state)."""
    lp = np.log(fit_df["price"] / fit_df["house_size"])
    state = float(lp.median())

    city_stats = lp.groupby(fit_df["city"]).agg(["median", "size"])
    city_val = _blend(city_stats, state, m)

    zip_stats = lp.groupby(fit_df["zip_code"]).agg(["median", "size"])
    zip_city = fit_df.groupby("zip_code")["city"].agg(lambda s: s.mode().iloc[0])
    zip_prior = zip_city.map(city_val).fillna(state)
    zip_val = _blend(zip_stats, zip_prior.reindex(zip_stats.index), m)

    return {"zip": zip_val, "city": city_val, "state": state}


def apply_zip_encoding(df: pd.DataFrame, enc: dict) -> pd.Series:
    """ZIP value if the ZIP was seen in training, else its city's value, else the state median."""
    out = df["zip_code"].map(enc["zip"]).astype(float)
    out = out.fillna(df["city"].map(enc["city"]).astype(float))
    return out.fillna(enc["state"])


def zip_encoding_out_of_fold(trn: pd.DataFrame, seed: int = SEED) -> pd.Series:
    """Training rows get values computed WITHOUT their own fold, so a home's own price
    never feeds its own feature (prevents target leakage)."""
    out = pd.Series(np.nan, index=trn.index)
    for fit_idx, enc_idx in KFold(N_FOLDS, shuffle=True, random_state=seed).split(trn):
        enc = fit_zip_encoding(trn.iloc[fit_idx])
        out.iloc[enc_idx] = apply_zip_encoding(trn.iloc[enc_idx], enc).values
    return out


# ---------------------------------------------------------------- features
def add_features(df: pd.DataFrame, lot_median: float, zip_feature: pd.Series) -> pd.DataFrame:
    out = df.copy()
    out["acre_lot_missing"] = out["acre_lot"].isna().astype(int)
    lot = out["acre_lot"].fillna(lot_median)
    out["log_acre_lot"] = np.log1p(lot)  # log1p because lots can be 0 acres
    out["log_house_size"] = np.log(out["house_size"])
    out["sold"] = (out["status"] == "sold").astype(int)
    out["zip_log_ppsf"] = zip_feature.values
    out["log_price"] = np.log(out["price"])
    return out[FEATURES + [TARGET] + META]


def build(save: bool = True) -> dict:
    """Run the full pipeline. Returns a dict with the three splits and a report."""
    nc, load_steps = load_nc_homes()
    clean, outlier_table = remove_outliers(nc)
    trn, val, tst = split(clean)

    lot_median = float(trn["acre_lot"].median())  # training data only
    enc_full = fit_zip_encoding(trn)
    data = {
        "train": add_features(trn, lot_median, zip_encoding_out_of_fold(trn)),
        "val": add_features(val, lot_median, apply_zip_encoding(val, enc_full)),
        "test": add_features(tst, lot_median, apply_zip_encoding(tst, enc_full)),
    }
    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        for name, d in data.items():
            d.to_csv(PROCESSED_DIR / f"{name}.csv", index=False)

    data["report"] = {
        "load_steps": load_steps,
        "outliers": outlier_table,
        "rows_after_outliers": len(clean),
        "lot_median": lot_median,
        "zip_encoding": enc_full,
    }
    return data


# ---------------------------------------------------------------- loading for models
def load_processed() -> dict:
    """Read the saved splits (run build() first). zip_code stays a 5-digit string."""
    return {
        name: pd.read_csv(PROCESSED_DIR / f"{name}.csv", dtype={"zip_code": "string"})
        for name in ["train", "val", "test"]
    }


class Standardize:
    """Standardize data with mean and std computed from TRAINING data (from the GD & LMS HW)."""

    def __init__(self):
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "Standardize":
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std


def get_arrays(data: dict | None = None, features: list[str] = FEATURES):
    """Standardized NumPy arrays for modeling.

    Returns X_trn, y_trn, X_val, y_val, X_tst, y_tst, scaler. Targets are log(price).
    """
    if data is None:
        data = load_processed()
    scaler = Standardize().fit(data["train"][features].to_numpy(float))
    arrays = []
    for name in ["train", "val", "test"]:
        X = scaler.transform(data[name][features].to_numpy(float))
        y = data[name][TARGET].to_numpy(float)
        arrays += [X, y]
    return (*arrays, scaler)


def main() -> None:
    data = build(save=True)
    rep = data["report"]
    print(rep["outliers"].to_string(index=False), "\n")
    print(f"Rows after outlier removal: {rep['rows_after_outliers']:,}")
    for name in ["train", "val", "test"]:
        print(f"{name:5s}: {len(data[name]):,} rows")
    print(f"Saved to {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
