# Project Notes: Predicting North Carolina Home Prices

Running log of decisions, reasons, and results. Write the report from this.

**Current project (since Nov 24):** predict the price of NC homes from the realtor.com "USA Real Estate" dataset. The project started as rent prediction with Craigslist data. Sections 2–3 below record that work and why we switched (section 5); it can go in the report as part of the data-selection story.

## Timeline
- Due: Wed Dec 2, 2026, 11:00 PM
- Target: code + results finished by **Sun Nov 29** (leaves Mon–Wed for writing)

| Date | Stage | Status |
|---|---|---|
| Nov 24 | Repo setup, data loading, choose dataset | done |
| Nov 24–25 | Exploration (EDA) | done Nov 25 |
| Nov 25–26 | Preprocessing | done Nov 25 |
| Nov 26–28 | Modeling + tuning | done Nov 25 |
| Nov 28–29 | Results analysis + investor application | done Nov 25 (v2 with $/sqft filter) |
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
- **Outlier limits (inclusive):** price $10k–$5M, size 300–15,000 sqft, beds ≤ 10, baths ≤ 10, lot ≤ 100 acres (missing lot sizes kept), **and price per sqft ≥ $40** (added after v1 results, see section 10).
  - Removed per rule (first rule hit): price 42, size 17, beds 31, baths 13, lot 40, **$/sqft 234** → **377 rows (0.92%)** → **40,390 homes**.
- **Split** (seed 42, 70/15/15): **train 28,273 / validation 6,058 / test 6,059**. Medians $332k–$340k; share sold 40–41%.
- **Lot size:** 5,382 missing, filled with the training median (0.35 acres) + `acre_lot_missing` flag. Missing-lot homes are cheaper/smaller ($305k / 1,497 sqft vs. $340k / 1,840 sqft).
- **ZIP feature `zip_log_ppsf`:** median log($/sqft) per ZIP, blended (n·ZIP + 10·city)/(n+10); city blended toward the state the same way; unseen ZIP → city → state.
  - Examples: 28207 Charlotte n=91: $397 → $373; 28212 Charlotte n=148: $186 → $187; 28642 Jonesville n=1: $93 → $162 (city $171).
  - **Out-of-fold (5-fold) encoding for training rows** prevents target leakage. One-feature linear fit: leaky train R² 0.285 / val 0.271; out-of-fold train 0.271 / val 0.271. Small here because smoothing already limits it; matters more for flexible models.
- **Final 7 inputs:** log_house_size, bed, bath, log_acre_lot (log1p), acre_lot_missing, sold, zip_log_ppsf. **Target:** log(price).
- **Dropped:** brokered_by, street (anonymized IDs), prev_sold_date (the sale date for sold homes → leaks status), state (constant), city/zip_code (captured by zip_log_ppsf).
- **Correlation with log price (train):** size 0.70, bath 0.66, ZIP 0.52, bed 0.47, lot 0.18, lot missing −0.07, sold −0.15. (Fig 10)
- **Standardized** with training mean/std (`Standardize` class from the GD & LMS HW). `get_arrays()` returns X/y for all splits.
- Processed CSVs in `data/processed/` (git-ignored; rebuild with `python src/preprocess.py`).

### 8. Modeling (Nov 25, `src/models.py`, `notebooks/03_modeling.ipynb`, figures fig11–fig17)
- **From-scratch implementations** adapted from homework: OLS (normal equation, pinv), Ridge OLS (bias not penalized), LMS mini-batch GD, KNN regressor (mean of k neighbors, chunked distances), K-means. Checked against scikit-learn: OLS and Ridge weights match to ~1e-14; KNN matches up to distance ties; K-means matches sklearn inertia.
- Two no-learning baselines: median price; ZIP $/sqft × size (rule of thumb).
- **Tuning (validation set), v2:**
  - GD: α ∈ {0.0003, 0.003, 0.03}, batch 64, 40 epochs → α=0.003 (max |w − w_OLS| = 0.003). α=0.0003 needs ~20 epochs. (Fig 12)
  - Poly + Ridge: degree 1/2/3 (7/35/119 features) × λ 1e-3 to 1e4 → **degree 3, λ=100**. Best val RMSE (log): deg1 0.305, deg2 0.299, deg3 0.296. The curve is flat for λ ≤ 100 (regularization barely matters with 28k rows) and rises past λ≈1,000. (Fig 13)
  - KNN: k from 1 to 100 → **k=15**. Val RMSE k=1 0.383 (overfit) → k=15 0.298 → k=100 0.308 (underfit). (Fig 14)
  - MLP (scikit-learn, ReLU, Adam, early stopping): layers {(32), (64,32), (128,64,32)} × alpha {1e-4, 1e-2, 1} → **(64,32), alpha=0.01**, 100 epochs, val RMSE 0.294. (Fig 15)
- **OLS coefficients (original units):** size elasticity 0.80 (1% bigger → +0.80%); ZIP $/sqft elasticity 1.06; +1 bath +9.9%; +1 bed −1.8% (holding sqft fixed); log lot +15.1% per unit; sold −7.6%; lot missing −3.9%. Standardized ranking: size 0.350, ZIP 0.307, bath 0.101, lot 0.068, sold −0.039, bed −0.017, lot missing −0.014. (Fig 11)

## Key results (v2: with the $40/sqft filter)
**Test set (6,059 homes, evaluated once):** (Figs 16, 17; `data/processed/test_results.csv`)

| Model | Setting | Median % err | Within 20% | MAE | RMSE ($) | R² (log) | R² ($) |
|---|---|---|---|---|---|---|---|
| Median price | none | 34.0% | 31.3% | $203,188 | $388,037 | 0.000 | −0.063 |
| ZIP $/sqft × size | none | 15.7% | 59.3% | $103,495 | $221,196 | 0.744 | 0.654 |
| OLS | none | 15.8% | 59.5% | $100,099 | $204,802 | 0.766 | 0.704 |
| GD (LMS) | α=0.003 | 15.8% | 59.5% | $100,121 | $204,798 | 0.767 | 0.704 |
| Poly + Ridge | deg 3, λ=100 | 15.7% | 60.1% | $98,783 | $205,304 | 0.774 | 0.702 |
| KNN | k=15 | **15.2%** | **60.8%** | $98,866 | $206,021 | 0.770 | 0.700 |
| MLP | (64,32), α=0.01 | 15.4% | 60.5% | **$97,492** | $204,505 | **0.781** | **0.705** |

- All models halve the median baseline's error (34% → ~15%), but beat the agent rule of thumb only slightly (15.7% → 15.2%; R² 0.744 → 0.781).
- Nonlinear models (MLP, Ridge, KNN) gain ~0.01–0.015 R² (log) and $1–2.6k MAE over OLS.
- Little overfitting: train ≈ validation for all models.
- Five very different models land within ~0.6 points → the limit is likely **missing features** (condition, age, exact location), not the algorithm.
- The gaps between top models (~0.3–0.6 pt) are about the size of split-to-split noise (section 10), so the ranking among them isn't decisive.
- Dollar RMSE (~$205k) is about 2× MAE (~$98k) because of a few expensive homes (section 9).

**v1 test results (no $/sqft filter, 6,094 test homes; kept for comparison, `test_results_v1_no_ppsf_filter.csv`):** median % error: median baseline 34.2, rule of thumb 15.8, OLS 15.7, GD 15.8, Poly+Ridge (deg 3, λ=0.001) 15.5, KNN (k=20) 15.0, MLP ((64,32), α=1e-4) 15.1. R² (log): 0.000 / 0.723 / 0.749 / 0.749 / 0.759 / 0.761 / 0.768.

### 9. Results analysis (Nov 25, `notebooks/04_results.ipynb`, figures fig18–fig20; main model = MLP, v2)
- **Overall:** median abs error 15.4%, median signed error −2.5%.
- **By actual price range** (median abs % / signed %): <$150k 36.2% / **+36.1%**; $150–250k 14.0% / +2.8%; **$250–400k 11.7% / −2.6%**; $400–700k 15.3% / −8.6%; $700k–1.5M 23.4% / −16.4%; >$1.5M 35.0% / **−33.3%**. → Regression toward the mean. (Fig 18)
- **Dollar error concentration:** top 5% priced homes (≥ $1,099,000) = 32% of absolute $ error, 74% of squared error; median error there 33.4% vs. 14.8%. Without them: MAE $69,895, RMSE $107,188 (vs. $97k / $205k).
- **By city** (median abs %): New Bern 10.8, Charlotte 13.2, Wilmington 13.2, Jacksonville 13.4, Raleigh 13.7, Leland 14.9, Fayetteville 15.7, Gastonia 15.7, Greenville 16.6, Asheville 19.2.
- **By status:** sold 14.9% vs. for_sale 15.7%.
- **K-means market segments** (own KMeans class; 392 ZIPs with ≥ 10 training homes; features: ZIP median log $/sqft, log size, log lot; elbow → k = 4) (Fig 19):

| Segment | ZIPs | Median $/sqft | Median sqft | Lot | Test error |
|---|---|---|---|---|---|
| Upscale, large homes | 102 | $204 | 2,319 | 0.35 ac | **13.3%** |
| Affordable | 150 | $144 | 1,659 | 0.38 ac | 14.5% |
| Pricey, compact homes | 79 | $240 | 1,658 | 0.35 ac | 17.7% |
| Rural, big lots | 61 | $181 | 1,692 | 1.13 ac | 21.5% |
| Small ZIP (< 10 homes) | n/a | n/a | n/a | n/a | 25.6% (+11.1% bias) |

- **Worst misses (v2):** listed at $41–$92/sqft (e.g., Robbinsville 1,140 sqft at $65,000, predicted $304k) → likely fixer-uppers or remaining non-market sales.
- **Investor screen** (3,654 for-sale test homes): ratio = predicted / asking. Naive ≥ 1.25: 602 flags (197 under $150k, driven by the cheap-home bias). Bias-adjusted (ratio ÷ median ratio of its price band ≥ 1.25): 628 flags; 160 with ratio ≥ 1.6 (median $134/sqft vs. $186 → likely condition/data issues); **468 plausible (1.25–1.6)**, 208 of them priced $150k–$400k. (Fig 20)
- **Improvements:** add year built/condition/coordinates/schools; separate model or weighting for the extremes; cross-validation / repeated splits for error bars; prediction intervals per home; current data or a price-index adjustment.

### 10. The $40/sqft filter: added after v1 test results (Nov 25)
- **Why:** in v1, all 10 worst test misses were priced $9–$32/sqft (e.g., 1,800 sqft sold for $15,950; Nags Head 2,314 sqft listed at $69,000). 38 v1 test homes were under $40/sqft, median error 245%. $40/sqft is below the 1st percentile (1st pct $50.6, 0.5th pct $37.4; typical ~$185). These look like non-market sales (auctions, partial interests, typos).
- **What:** added `MIN_PRICE_PER_SQFT = 40` to preprocessing (234 extra rows removed, 0.58%), retrained and re-tuned every model, and re-evaluated on the new test set.
- **Honesty note for the report:** the rule was chosen *after* seeing test errors, so the report should say so. The v1 results are kept for comparison.
- **Result (MLP):**

| Version | Test homes | Median % err | Within 20% | MAE | R² (log) |
|---|---|---|---|---|---|
| v1 (no filter) | 6,094 | 15.1% | 61.1% | $93,084 | 0.768 |
| v1 model, under-$40 homes removed from its test set | 6,056 | **14.9%** | 61.5% | $92,752 | 0.783 |
| v2 (filter, retrained) | 6,059 | 15.4% | 60.5% | $97,492 | 0.781 |

- **Interpretation:** the filter did not make the model better. Removing 0.58% of rows barely changes what the model learns, and v2's slightly worse number comes from a different random test sample (rows removed before the split change which homes land in test). **Lesson: changing the test sample moves the score by ~0.3–0.5 points, about as much as the gaps between the top models.** Cross-validation or repeated splits would give error bars.
- Data cleaning is still justified: the v2 data no longer contains prices that aren't market values, which matters for the investor screen.

## Sources to cite (MLA 9; verify each one yourself before citing)
Status key: ✔ = checked online Nov 25 (journal, volume, pages); ● = standard citation I'm confident in, but still open it and confirm; ⚠ = check the details noted.

**Data**
- ⚠ Sakib, Ahmed Shahriar. "USA Real Estate Dataset." *Kaggle*, www.kaggle.com/datasets/ahmedshahriarsakib/usa-real-estate-dataset. Accessed 24 Nov. 2026.
  - Check on the Kaggle page: the author's name as displayed, the version/update date, the license, and the stated collection period. Use for: Data section, in-text (Sakib).
- ⚠ Reese, Austin. "USA Housing Listings." *Kaggle*, www.kaggle.com/datasets/austinreese/usa-housing-listings. Accessed 24 Nov. 2026.
  - Only if you mention the first dataset considered (Craigslist rentals). Check the author name.

**Related work (housing prices)**
- ✔ Rosen, Sherwin. "Hedonic Prices and Implicit Markets: Product Differentiation in Pure Competition." *Journal of Political Economy*, vol. 82, no. 1, 1974, pp. 34–55.
  - The classic idea behind this project: a home's price is the sum of implicit prices of its characteristics (size, baths, location). In-text: (Rosen).
- ✔ De Cock, Dean. "Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester Regression Project." *Journal of Statistics Education*, vol. 19, no. 3, 2011.
  - Well-known housing-price regression dataset; shows the task is standard for ML coursework. Ames has ~80 features including quality and year built, which our data lacks (useful for the "missing features" discussion). In-text: (De Cock).
- ● Harrison, David, Jr., and Daniel L. Rubinfeld. "Hedonic Housing Prices and the Demand for Clean Air." *Journal of Environmental Economics and Management*, vol. 5, no. 1, 1978, pp. 81–102.
  - Origin of the Boston Housing dataset. Optional.
- ⚠ Boeing, Geoff, and Paul Waddell. "New Insights into Rental Housing Markets across the United States: Web Scraping and Analyzing Craigslist Rental Listings." *Journal of Planning Education and Research*, vol. 37, no. 4, 2017, pp. 457–476.
  - Only if you discuss the Craigslist rental data. Confirm volume/pages.

**Methods**
- ✔ Micci-Barreca, Daniele. "A Preprocessing Scheme for High-Cardinality Categorical Attributes in Classification and Prediction Problems." *ACM SIGKDD Explorations Newsletter*, vol. 3, no. 1, 2001, pp. 27–32, doi:10.1145/507533.507538.
  - The original target-encoding paper; it describes blending ZIP-code statistics across hierarchy levels, exactly our ZIP → city → state smoothing. In-text: (Micci-Barreca).
- ● Hoerl, Arthur E., and Robert W. Kennard. "Ridge Regression: Biased Estimation for Nonorthogonal Problems." *Technometrics*, vol. 12, no. 1, 1970, pp. 55–67.
  - Ridge regression. In-text: (Hoerl and Kennard).
- ● Kingma, Diederik P., and Jimmy Ba. "Adam: A Method for Stochastic Optimization." *3rd International Conference on Learning Representations (ICLR)*, 2015, arxiv.org/abs/1412.6980.
  - The optimizer MLPRegressor uses. In-text: (Kingma and Ba).

**Software**
- ● Pedregosa, Fabian, et al. "Scikit-learn: Machine Learning in Python." *Journal of Machine Learning Research*, vol. 12, 2011, pp. 2825–2830. In-text: (Pedregosa et al.)
- ● Harris, Charles R., et al. "Array Programming with NumPy." *Nature*, vol. 585, 2020, pp. 357–362, doi:10.1038/s41586-020-2649-2.
- ● McKinney, Wes. "Data Structures for Statistical Computing in Python." *Proceedings of the 9th Python in Science Conference*, 2010, pp. 56–61. (pandas)
- ● Hunter, John D. "Matplotlib: A 2D Graphics Environment." *Computing in Science & Engineering*, vol. 9, no. 3, 2007, pp. 90–95.
- ● Waskom, Michael L. "seaborn: Statistical Data Visualization." *Journal of Open Source Software*, vol. 6, no. 60, 2021, p. 3021.

**Course materials and AI** (templates; fill in the real details)
- Course notes: Lastname, Firstname. "Lecture title." ITCS 3156: Introduction to Machine Learning, University of North Carolina at Charlotte, Fall 2026. Lecture slides/notes.
  - Use for the OLS, gradient descent, Ridge, KNN, K-means and neural-network formulas. Check the instructor's exact name on Canvas.
- AI tool (MLA's recommended format): "Description or text of the prompt" prompt. *Claude*, Opus 5.5 version, Anthropic, 25 Nov. 2026, claude.ai.
  - MLA treats the prompt as the title. Also describe the AI use in the Acknowledgement section (follow the course's "AI Acknowledgement Example" file).

**Where to cite in the report**
- Data: Sakib (dataset), De Cock (comparison).
- Introduction / related work: Rosen (hedonic pricing), De Cock / Harrison and Rubinfeld (housing regression in ML).
- Methods: course notes (OLS, GD, Ridge, KNN, K-means, MLP), Hoerl and Kennard (Ridge), Micci-Barreca (ZIP target encoding), Kingma and Ba (Adam).
- Implementation: Pedregosa et al., Harris et al., McKinney, Hunter, Waskom.
