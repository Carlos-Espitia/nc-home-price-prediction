# Predicting North Carolina Home Prices with Machine Learning

ITSC 3156 (Introduction to Machine Learning) final project by Carlos Espitia, UNC Charlotte, Fall 2026.

This project predicts the price of North Carolina homes from their size, bedrooms, bathrooms, lot size, sale status and location (ZIP code). It compares five models from the course: linear regression solved with OLS and with gradient descent, polynomial + Ridge regression, KNN regression, and a neural network. It also uses K-means to segment NC housing markets. Finally, the model is used to screen for-sale listings priced below what similar homes go for.

## Results at a glance
**Dataset: 40,390 NC homes** after cleaning, with 7 input features, split 70/15/15 into 28,273 training, 6,058 validation and 6,059 test homes. The table below is on the **test set**: homes never used for training or tuning.

| Model | Median % error | Homes within ±20% | R² (log price) |
|---|---|---|---|
| Guess the median price (baseline) | 34.0% | 31.3% | 0.00 |
| ZIP $/sqft × house size (rule of thumb) | 15.7% | 59.3% | 0.74 |
| Linear regression (OLS) | 15.8% | 59.5% | 0.77 |
| Linear regression (gradient descent) | 15.8% | 59.5% | 0.77 |
| Polynomial (degree 3) + Ridge | 15.7% | 60.1% | 0.77 |
| KNN regression (k = 15) | **15.2%** | **60.8%** | 0.77 |
| Neural network (64, 32) | 15.4% | 60.5% | **0.78** |

Key findings:
- Every model halves the error of guessing the median, but only slightly beats the rule of thumb "ZIP price per sqft × house size". Size and location carry most of the signal.
- The model is most accurate for $250k–$400k homes (11.7% median error). It over-predicts cheap homes and under-predicts expensive ones (regression toward the mean).
- Five very different models land within about 0.6 points of each other, which suggests missing information (condition, age, exact location) limits accuracy more than the choice of algorithm.

## Project structure
```
src/
  load_data.py        load the Kaggle CSV, filter to NC homes, remove duplicates
  preprocess.py       outliers, train/val/test split, ZIP target encoding, log transforms, standardization
  models.py           from-scratch OLS, Ridge, gradient descent (LMS), KNN regression, K-means
  plot_style.py       shared figure style
notebooks/
  01_eda.ipynb             data exploration (figures 1-9)
  02_preprocessing.ipynb   preprocessing steps explained (figure 10)
  03_modeling.ipynb        training, tuning and test evaluation (figures 11-17)
  04_results.ipynb         error analysis, K-means segments, investor screen (figures 18-20)
figures/              all report figures (PNG)
data/                 raw and processed data (not committed; see below)
NOTES.md              decision log with every result and the reasoning behind it
COURSE_TOPICS.md      course topics and how the project uses them
```

## Setup
Requires Python 3.12+ (developed on 3.14).
```bash
python -m venv .venv
# Windows:   .venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Getting the data
Dataset: "USA Real Estate Dataset" (realtor.com listings) on Kaggle:
https://www.kaggle.com/datasets/ahmedshahriarsakib/usa-real-estate-dataset

Download the zip from that page, extract the CSV, and save it as `data/raw/realtor-data.csv`. The file inside the zip is named `realtor-data.zip.csv`; rename it.

Or use the Kaggle API (token at `~/.kaggle/kaggle.json`):
```bash
kaggle datasets download -d ahmedshahriarsakib/usa-real-estate-dataset -p data/raw --unzip
```
Then rename the extracted CSV to `data/raw/realtor-data.csv`.

## Running the pipeline
```bash
python src/load_data.py      # optional: summary of the NC filtering steps
python src/preprocess.py     # builds data/processed/{train,val,test}.csv
```
Then run the notebooks in order (01 → 04) from the `notebooks/` folder. Notebook 03 takes about 2–3 minutes; the others take under a minute. All randomness uses fixed seeds, so results are reproducible.

## Data notes
- About 40,400 NC homes (for sale or sold) after removing land lots, duplicates, impossible values and non-market sales under $40/sqft.
- The sold homes sold between Oct 2021 and Apr 2022, so prices reflect that period.
- The dataset has no coordinates, so location is captured at the ZIP-code level.
