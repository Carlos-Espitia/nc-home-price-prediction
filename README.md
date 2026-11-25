# Predicting North Carolina Home Prices with Machine Learning

ITSC 3156 final project by Carlos Espitia (UNC Charlotte).

Predicts the price of North Carolina homes from their features (beds, baths, house size, lot size, location), comparing several models from the course. The model is then used to flag homes listed below their predicted value as possible investment opportunities.

## Project structure
```
data/raw/        raw Kaggle CSVs (not committed; see below)
data/processed/  cleaned data produced by the pipeline
notebooks/       exploration and analysis notebooks
src/             reusable Python code
figures/         report figures (PNG)
NOTES.md         decision log
COURSE_TOPICS.md course methods and how the project uses them
```

## Setup
```bash
python -m venv .venv
# Windows:   .venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Getting the data
Dataset: "USA Real Estate Dataset" (realtor.com listings) on Kaggle, https://www.kaggle.com/datasets/ahmedshahriarsakib/usa-real-estate-dataset

Download the zip from that page, extract the CSV, and save it as `data/raw/realtor-data.csv`. The file inside the zip is named `realtor-data.zip.csv`; rename it.

Or use the Kaggle API (token at `~/.kaggle/kaggle.json`):
```bash
kaggle datasets download -d ahmedshahriarsakib/usa-real-estate-dataset -p data/raw --unzip
```

Check that the data loads:
```bash
python src/load_data.py
```
