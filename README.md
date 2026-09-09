# E-Commerce Churn Prediction + Constrained Retention Budget Optimization

Predicts which customers will stop purchasing in the next 60 days, explains *why*
with SHAP, and — going beyond a typical churn project — turns those risk scores into
an actual budget-allocation **decision** using linear programming: given a fixed
retention budget, which customers get which retention action (email / discount /
personal outreach) to maximize expected customer-lifetime-value saved?

**Headline result:** the LP-optimized allocation saves **66.7% more expected CLV**
than the standard "target the top-N riskiest customers" heuristic, for the same
budget. See [`report/BUSINESS_REPORT.md`](report/BUSINESS_REPORT.md) for the full
writeup with numbers.

## Pipeline

| Step | Script | Output |
|---|---|---|
| 1. Generate data | `src/01_generate_data.py` | `data/raw/*.csv` |
| 2. Feature engineering | `src/02_feature_engineering.py` | `data/processed/customer_features.csv` |
| 3. EDA | `src/03_eda.py` | `outputs/eda/*.png` |
| 4. Model training & comparison | `src/04_model_training.py` | `outputs/models/model_comparison.csv`, `best_model.pkl` |
| 5. SHAP explainability | `src/05_explainability.py` | `outputs/models/shap_*.png/csv` |
| 6. Retention budget optimization (LP) | `src/06_budget_optimization.py` | `outputs/optimization/*.csv` |
| 7. Power BI exports | `src/07_powerbi_exports.py` | `powerbi/*.csv` |

Run in order:
```bash
pip install -r requirements.txt
python src/01_generate_data.py
python src/02_feature_engineering.py
python src/03_eda.py
python src/04_model_training.py
python src/05_explainability.py
python src/06_budget_optimization.py
python src/07_powerbi_exports.py
```

## Why synthetic data
No single public dataset combines purchase history, returns, discounts, reviews,
and session activity at customer level (UCI Online Retail has transactions only;
Olist and Kaggle "ecommerce customer" sets each cover a subset). `01_generate_data.py`
documents an explicit data-generating process — a latent engagement persona plus a
randomized "churn shock" that decays purchase rate and depresses reviews/inflates
returns after it hits — so the churn label is a noisy, multi-cause outcome the model
has to actually discover, not a variable we hand-coded.

## What makes this different from a standard churn project
Most churn portfolio projects stop at "Random Forest got 91% accuracy" and a bullet
list of segments. This one:
1. Uses a proper time-based feature cutoff to avoid label leakage (60-day
   forward-looking prediction, not same-period features).
2. Explains predictions with SHAP, and reports a **negative finding** (discount
   usage does *not* predict retention) rather than only positive-sounding results.
3. Converts risk scores into a **constrained optimization problem** (PuLP linear
   program: maximize expected CLV saved subject to budget + staffing capacity),
   and benchmarks it against the naive heuristic with a real, computed % improvement.

## Stack
Python (Pandas, NumPy, scikit-learn, XGBoost, SHAP, PuLP, Matplotlib/Seaborn),
Power BI (see `powerbi/POWERBI_GUIDE.md` for the 4-page dashboard build).

## Repo structure
```
churn_project/
├── data/{raw,processed}/
├── src/                  01_generate_data.py ... 07_powerbi_exports.py
├── outputs/{eda,models,optimization}/
├── powerbi/               CSVs + POWERBI_GUIDE.md
├── report/BUSINESS_REPORT.md
└── requirements.txt
```
