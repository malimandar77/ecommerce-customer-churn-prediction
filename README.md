# E-Commerce Customer Churn Prediction

An individual AI & Data Science II mini-project that predicts whether an e-commerce customer is likely to churn using transaction-history features.

## Problem
Customer churn means a customer stops purchasing. The system learns customer behaviour from historical transactions and predicts churn risk for a future period.

## Dataset
UCI Online Retail dataset: https://archive.ics.uci.edu/dataset/352/online+retail
The dataset contains transactions from 01/12/2010 to 09/12/2011 for a UK-based non-store online retailer.

## Pipeline
1. Download UCI transaction data
2. Clean cancellations, missing customers and invalid quantities/prices
3. Create a time-based customer snapshot
4. Engineer RFM-style behavioural features
5. Create a forward-looking churn label
6. Train Logistic Regression and Random Forest
7. Compare F1 and ROC-AUC
8. Save the best model
9. Run Streamlit UI

## Important methodological choice
The churn label is NOT defined from the same period as the input features. Features are calculated before a snapshot date, while churn is determined from the following 90 days. This reduces target leakage.

## Project structure
- `src/download_data.py` — downloads UCI data
- `src/train.py` — preprocessing, feature engineering, training and evaluation
- `app.py` — Streamlit prediction interface
- `notebooks/EDA_Model.ipynb` — EDA/model walkthrough
- `requirements.txt` — dependencies

## Local setup

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

Then:
```bash
pip install -r requirements.txt
python src/download_data.py
python src/train.py
streamlit run app.py
```

## Deployment
Push the repository to GitHub and deploy `app.py` through Streamlit Community Cloud.

## Academic note
This project is intended as an educational demonstration. It is not a production customer-retention system.
