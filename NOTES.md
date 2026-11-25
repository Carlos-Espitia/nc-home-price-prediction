# Project Notes: Predicting North Carolina Home Prices

Running log of decisions, reasons, and results. Write the report from this.

**Current project (since Nov 24):** predict the price of NC homes from the realtor.com "USA Real Estate" dataset. The project started as rent prediction with Craigslist data. Sections 2–3 below record that work and why we switched (section 5); it can go in the report as part of the data-selection story.

## Timeline
- Due: Wed Dec 2, 2026, 11:00 PM
- Target: code + results finished by **Sun Nov 29** (leaves Mon–Wed for writing)

| Date | Stage | Status |
|---|---|---|
| Nov 24 | Repo setup, data loading, choose dataset | done |
| Nov 24–25 | Exploration (EDA) | |
| Nov 25–26 | Preprocessing | |
| Nov 26–28 | Modeling + tuning | |
| Nov 28–29 | Results analysis + investor application | |
| Nov 30–Dec 2 | Write report | |

## Plan (proposed Nov 24; see COURSE_TOPICS.md for the full mapping)
**Task:** supervised **regression**: given a home's features, predict its price (USD).

**Features available:** `bed`, `bath`, `house_size` (sqft), `acre_lot`, `city`, `zip_code`, `status`. Probably dropped: `street` and `brokered_by` (anonymized ID numbers, no meaning), `prev_sold_date` (impossible values, mostly missing).

**Models (all from class):**
1. Linear regression with OLS, from scratch (HW: Derivatives & OLS). Baseline; coefficients give "$ per sqft / per bedroom".
2. Linear regression with gradient descent (LMS), from scratch (HW: GD & LMS). Same model trained a different way; learning curve.
3. Polynomial features + Ridge (HW: Curve Fitting & Regularization). Tests curved effects and overfitting; tune lambda on validation data.
4. KNN regression (HW: KNN, changed from voting to averaging). "Comparable homes" pricing; tune k.
5. Neural network, `MLPRegressor` (lectures + ML sketch). Flexible nonlinear model.
6. Supporting analysis: K-means (HW: K-means) to segment NC markets; PCA optional.

**Metrics:** RMSE, MAE (in dollars), R².
**Investor angle:** homes listed well below the model's predicted price = possible deals.

**Decided (Nov 25, Carlos):**
- **Target: both `for_sale` and `sold` homes**, with `status` as a feature so the model can learn the list-vs-sold difference. In Results, report error separately for each group. `ready_to_build` is excluded, since those are mostly land or homes not yet built.
- **Drop land lots:** remove rows missing bed, bath or house_size.
- **Keep the realtor.com dataset (Nov 25).** Looked for NC data with exact coordinates. NC OneMap parcels have coordinates but only tax assessed values (not sale prices) and no beds/baths/sqft. King County (Seattle, 2014–15) has coordinates but isn't NC and is older. Joining county sale records to parcels wouldn't fit the timeline. Possible future work: join NC OneMap parcel centroids for exact locations.
- **Location (decided Nov 25):** the main location feature is the **ZIP's median price per sqft**, computed from the training set only (target encoding) and blended toward the city/state value for small ZIPs. ZIP coordinates are optional, for an EDA map only.
  - Why $/sqft instead of median price: prices inside one ZIP vary a lot, mostly because of size. Measured Nov 25 on ZIPs with 30+ homes: the typical within-ZIP spread (std/mean) is **0.60 for price but 0.33 for $/sqft**. Example: ZIP 28207 runs from $217k to $2.5M (10th–90th percentile), but $/sqft only from $268 to $603.
  - The ZIP median price alone explains roughly 37% of log-price variation (rough, in-sample). Size, beds and baths must explain much of the rest.
  - The dataset has no coordinates, and `street` is an anonymized ID, so homes can't be geocoded. ZIP is the finest location available, which is a limitation for the report.

**Still open:**
- Log-transform the price (likely, as in the Data Exploration HW).
- Outlier cutoffs (e.g. $0 prices, 800,900 sqft, 90 beds).

## Decisions log

### 1. Repo structure (Nov 24)
- `data/` is git-ignored because the raw CSVs are far larger than GitHub's 100 MB per-file limit. The README explains how to download the data instead.
- Used a virtual environment (`.venv`) so the project's package versions don't clash with other Python work and `requirements.txt` fully describes the environment.

### 2. First dataset considered: Craigslist rentals (no longer used)
- Source: "USA Housing Listings" by Austin Reese on Kaggle, https://www.kaggle.com/datasets/austinreese/usa-housing-listings. File `data/raw/housing.csv` (558 MB), kept in case it's needed.
- 384,977 rows × 22 columns. Target would have been `price` (monthly rent).
- Missing values: `parking_options` 37%, `laundry_options` 21%. Junk prices: NC rent ranged from $0 to $11.6M.
- Data scraped around 2020.
- Searched for newer public rental datasets (Nov 24). Found nothing newer that is free and listing-level. The Kaggle uploads dated 2024 appear to be re-uploads of the 2019 UCI "Apartment for Rent Classified" data. Current listing data is commercial only.

### 3. Craigslist duplicates (why NC rentals were too small)
- Over half the rows were reposts (identical price, type, sqft, beds, baths, lat, long).

| Scope | Raw rows | After dedup |
|---|---|---|
| NC | 18,628 | 7,535 (below the 10k requirement) |
| Southeast (12 states) | 126,794 | 50,485 |
| All US | 384,977 | 179,921 |

- Lesson that also applies to the realtor data: remove duplicates before the train/test split. Otherwise copies of a listing land in both sets and test scores look better than they really are (data leakage).

### 4. Current dataset: USA Real Estate (realtor.com)
- https://www.kaggle.com/datasets/ahmedshahriarsakib/usa-real-estate-dataset. File `data/raw/realtor-data.csv`; the original zip is saved as `usa_real_estate_archive.zip`.
- 2,226,382 rows × 12 columns: brokered_by, status, price, bed, bath, acre_lot, street, city, state, zip_code, house_size, prev_sold_date.
- `price` is the list/sale price. No lat/long, no property-type column, no amenities.
- `status`: for_sale 1.39M, sold 812k, ready_to_build 25k.
- NC: 85,745 rows, 81,628 after dedup. **40,767 have price + bed + bath + house_size** (24,462 for_sale, 16,305 sold). The other ~40k are mostly missing bed/bath/size and are likely land lots.
- NC prices: median $249,900; range $0 to $37.5M. House size up to 800,900 sqft and acre_lot up to 100,000 are clearly errors, so preprocessing will need outlier rules.
- Top NC cities: Charlotte 6,875, Winston-Salem 2,362, Wilmington 1,646, Greensboro 1,596, Jacksonville 1,579.
- Data quality: `street` and `brokered_by` are anonymized numbers. `prev_sold_date` includes impossible years (1901 to 3019).
- There's no column saying when each home was listed. Check the Kaggle page for the collection date and quote it in the report.

### 5. Switched to predicting home prices (Nov 24, Carlos's choice)
- Reason: the rental data is older (about 2020) than the realtor.com data, and the realtor data has more than 10k NC samples by itself (~40.8k usable). NC rentals had only 7.5k after dedup.
- Trade-off: the project no longer predicts rent, but the purchase price is half of any rental-investment decision, so the investor angle carries over.

### 6. Exploration results (Nov 25, `notebooks/01_eda.ipynb`, figures fig01–fig09)
- **Filtering funnel:** 2,226,382 US → 85,745 NC → 84,591 for_sale/sold → 42,946 with bed/bath/size → **40,767 after dedup** (24,462 for sale, 16,305 sold).
- **Data time period:** all sold homes sold **Oct 26, 2021 to Apr 26, 2022**. The latest date on any listing is June 9, 2022, so this is a ~mid-2022 snapshot, only about 1–2 years newer than the Craigslist data (about 2020). The main reason the switch still holds is the sample size: 40.8k NC homes vs. 7.5k rentals. Limitation for the report: NC prices have changed since 2022.
- **Missing:** prev_sold_date 29.2%, acre_lot 13.3%, brokered_by 1.0%, street 8 rows, zip 1 row.
  - For sold homes, `prev_sold_date` is the sale date itself, so it can't be used as a feature (it marks a home as sold, which leaks the status).
- **Skew:** price skewness 7.75 → **-0.11 after log**. Mean $430,748 vs. median $335,000. → Predict log(price). (Fig 1)
- **Impossible values** (Fig 2): price < $10k: 4 rows; > $5M: 38; size < 300 sqft: 9; > 15,000 sqft: 11; beds > 10: 39; baths > 10: 43; lot > 100 acres: 45. Max values: 90 beds, 54 baths, 800,900 sqft, 100,000 acres.
- **Correlations** (Fig 8): raw price vs. size **0.17** (wrecked by outliers); log price vs. log size **0.70**. Log price with: bath 0.64, bed 0.42, log lot 0.19, sold -0.14. Size–bath 0.74, bed–bath 0.69, size–bed 0.62 → strong overlap (multicollinearity) → motivates Ridge.
- **Beds/baths** (Fig 5): median price rises steadily, e.g. 3 bed $300k → 4 bed $422k → 5 bed $575k; 2 bath $265k → 3 bath $375k → 4 bath $590k.
- **Location** (Fig 6): median $/sqft Fayetteville $122, Greenville $125, Jacksonville $128, New Bern $141, Gastonia $163, Leland $184, Wilmington $205, Charlotte $213, Raleigh $229, Asheville $287. ZIP medians range from $16 to $1,290/sqft. 727 ZIPs, median 17 homes each, 281 (39%) with < 10 homes. (Fig 9)
- **Status** (Fig 7): for_sale median $353,720 / 1,891 sqft vs. sold $300,000 / 1,664 sqft. Part of the gap is size.

### 7. Preprocessing (Nov 25, approved by Carlos; `src/preprocess.py`, `notebooks/02_preprocessing.ipynb`)
- **Outlier limits (inclusive):** price $10k–$5M, size 300–15,000 sqft, beds ≤ 10, baths ≤ 10, lot ≤ 100 acres (missing lot sizes kept). Removed **143 rows (0.35%)** → **40,624**. Removed per rule (first rule hit): price 42, size 17, beds 31, baths 13, lot 40.
- **Split** (seed 42): **train 28,436 / validation 6,094 / test 6,094** (70/15/15). Similar medians ($330k–$335k) and share sold (40%).
- **Lot size:** 5,415 missing, filled with the training median (0.35 acres) + `acre_lot_missing` flag. Missing-lot homes are cheaper/smaller ($303k / 1,497 sqft vs. $339k / 1,837 sqft).
- **ZIP feature `zip_log_ppsf`:** median log($/sqft) per ZIP, blended (n·ZIP + 10·city)/(n+10); city blended toward the state the same way; unseen ZIP → city → state.
  - Examples: 28207 Charlotte n=93: $402 → $378; 28212 Charlotte n=145: $187 → $189; 27101 Winston-Salem n=1: $83 → $142 (city $150).
  - **Out-of-fold (5-fold) encoding for training rows** prevents target leakage. Experiment with a one-feature linear fit: leaky train R² 0.289 / val 0.277; out-of-fold train 0.275 / val 0.277. Small effect here because smoothing already limits it; matters more for flexible models (KNN, MLP).
- **Final 7 inputs:** log_house_size, bed, bath, log_acre_lot (log1p), acre_lot_missing, sold, zip_log_ppsf. **Target:** log(price).
- **Dropped:** brokered_by, street (anonymized IDs), prev_sold_date (the sale date for sold homes → leaks status), state (constant), city/zip_code (captured by zip_log_ppsf).
- **Correlation with log price (train):** size 0.70, bath 0.66, ZIP 0.52, bed 0.46, lot 0.18, lot missing -0.07, sold -0.14. (Fig 10)
- **Standardized** with training mean/std (`Standardize` class from the GD & LMS HW). `get_arrays()` returns X/y for all splits.
- Processed CSVs in `data/processed/` (git-ignored; rebuild with `python src/preprocess.py`).

### 8. Modeling (Nov 25, `src/models.py`, `notebooks/03_modeling.ipynb`, figures fig11–fig17)
- **From-scratch implementations** adapted from homework: OLS (normal equation, pinv), Ridge OLS (bias not penalized), LMS mini-batch GD, KNN regressor (mean of k neighbors, chunked distances). Checked against scikit-learn: OLS and Ridge weights match to ~1e-14; KNN matches up to distance ties.
- Two no-learning baselines added: median price; ZIP $/sqft × size (rule of thumb).
- **Tuning (validation set):**
  - GD: α ∈ {0.0003, 0.003, 0.03}, batch 64, 40 epochs → α=0.003 (max |w − w_OLS| = 0.0035). (Fig 12)
  - Poly + Ridge: degree 1/2/3 (7/35/119 features) × λ 1e-3 to 1e4 → **degree 3, λ=0.001** (smallest λ tried: with 28k rows, regularization barely matters). Val RMSE (log): deg1 0.322, deg2 0.314, deg3 0.311. (Fig 13)
  - KNN: k from 1 to 100 → **k=20**. Val RMSE k=1 0.427 (overfit) → k=20 0.316 → k=100 0.327 (underfit). (Fig 14)
  - MLP (scikit-learn, ReLU, Adam, early stopping): layers {(32), (64,32), (128,64,32)} × alpha {1e-4, 1e-2, 1} → **(64,32), alpha=1e-4**, 124 epochs, val RMSE 0.311. (Fig 15)
- **OLS coefficients (original units):** size elasticity 0.80 (1% bigger → +0.80%); ZIP $/sqft elasticity 1.07; +1 bath +10.7%; +1 bed −2.3% (holding sqft fixed); log lot +14.2% per unit; sold −7.4%; lot missing −4.5%. Standardized ranking: size 0.354, ZIP 0.314, bath 0.110, lot 0.065, sold −0.038, bed −0.021, lot missing −0.016. (Fig 11)

## Key results
**Test set (6,094 homes, evaluated once):** (Figs 16, 17; `data/processed/test_results.csv`)

| Model | Median % err | Within 20% | MAE | RMSE ($) | R² (log) | R² ($) |
|---|---|---|---|---|---|---|
| Median price | 34.2% | 30.7% | $196,603 | $379,009 | 0.000 | −0.051 |
| ZIP $/sqft × size | 15.8% | 59.4% | $99,079 | $213,468 | 0.723 | 0.667 |
| OLS | 15.7% | 60.1% | $95,319 | $192,905 | 0.749 | 0.728 |
| GD (LMS) | 15.8% | 60.0% | $95,533 | $192,600 | 0.749 | 0.729 |
| Poly + Ridge | 15.5% | 60.6% | $93,892 | $195,340 | 0.759 | 0.721 |
| KNN (k=20) | **15.0%** | **61.7%** | $95,148 | $204,624 | 0.761 | 0.694 |
| MLP (64,32) | 15.1% | 61.1% | **$93,084** | $194,422 | **0.768** | 0.723 |

- All models beat the median baseline by 2×, but beat the agent rule of thumb only slightly (15.8% → 15.0%).
- Nonlinear models (MLP, KNN, poly) gain ~0.5 pt median error / ~0.01–0.02 R² over OLS.
- Little overfitting: train ≈ validation for OLS/Ridge/MLP.
- Five very different models land within ~1 point → the limit is likely **missing features** (condition, age, exact location), not the algorithm.
- Dollar RMSE (~$190k) is far above MAE (~$94k) → a few expensive homes produce huge dollar misses. Look at this in error analysis.

## Sources to cite (verify each one yourself)
- USA Real Estate Dataset, Kaggle (Ahmed Shahriar Sakib), https://www.kaggle.com/datasets/ahmedshahriarsakib/usa-real-estate-dataset
- scikit-learn (Pedregosa et al., 2011, *Journal of Machine Learning Research*)
- (more to add)
