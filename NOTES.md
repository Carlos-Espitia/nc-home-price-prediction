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

## Key results
(fill in as we go)

## Sources to cite (verify each one yourself)
- USA Real Estate Dataset, Kaggle (Ahmed Shahriar Sakib), https://www.kaggle.com/datasets/ahmedshahriarsakib/usa-real-estate-dataset
- scikit-learn (Pedregosa et al., 2011, *Journal of Machine Learning Research*)
- (more to add)
