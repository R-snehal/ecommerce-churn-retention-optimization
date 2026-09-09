"""
02_feature_engineering.py
--------------------------
Builds the customer-level feature table used for EDA and modelling.

Churn definition (business rule, stated explicitly - this is a modelling choice,
not ground truth):
  A customer is labelled CHURNED (target = 1) if they made NO purchase in the
  60 days immediately before the "as-of" date (2025-08-31), i.e. they are NOT
  expected to purchase again in the next 30-60 days, conditional on their
  history up to 60 days before as-of. To avoid leaking the label into the
  features, all recency/frequency/value features are computed using only
  transactions up to (as_of - 60 days) - a proper "feature cutoff" - and the
  label is whether the customer purchased in the 60-day window AFTER that cutoff.
  This mirrors a real 60-day forward-looking churn prediction task.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

RAW = "/home/claude/churn_project/data/raw"
OUT = "/home/claude/churn_project/data/processed"
import os
os.makedirs(OUT, exist_ok=True)

AS_OF = datetime(2025, 8, 31)
PREDICTION_WINDOW_DAYS = 60
FEATURE_CUTOFF = AS_OF - timedelta(days=PREDICTION_WINDOW_DAYS)   # 2025-07-02

customers = pd.read_csv(f"{RAW}/customers.csv", parse_dates=["signup_date"])
tx = pd.read_csv(f"{RAW}/transactions.csv", parse_dates=["order_date"])
reviews = pd.read_csv(f"{RAW}/reviews.csv", parse_dates=["review_date"])
sessions = pd.read_csv(f"{RAW}/sessions.csv", parse_dates=["month"])

# ---- split transactions at the feature cutoff ----
tx_hist = tx[tx.order_date <= FEATURE_CUTOFF].copy()          # used to build features
tx_future = tx[(tx.order_date > FEATURE_CUTOFF) & (tx.order_date <= AS_OF)].copy()  # used only for label

# Only customers who had signed up by the cutoff and had a chance to be observed
eligible = customers[customers.signup_date <= FEATURE_CUTOFF].copy()

# ---------------------------------------------------------------------------
# Core RFM + engineered features (computed on tx_hist only)
# ---------------------------------------------------------------------------
grp = tx_hist.groupby("customer_id")

feat = grp.agg(
    total_orders=("order_id", "count"),
    total_spend=("net_amount", "sum"),
    average_order_value=("net_amount", "mean"),
    first_purchase=("order_date", "min"),
    last_purchase=("order_date", "max"),
    total_returns=("is_returned", "sum"),
    avg_discount_pct=("discount_pct", "mean"),
    discount_orders=("discount_pct", lambda x: (x > 0).sum()),
    distinct_categories=("category", "nunique"),
).reset_index()

feat["days_since_last_purchase"] = (FEATURE_CUTOFF - feat["last_purchase"]).dt.days
feat["customer_span_days"] = (feat["last_purchase"] - feat["first_purchase"]).dt.days.clip(lower=1)
feat["purchase_frequency"] = feat["total_orders"] / (feat["customer_span_days"] / 30.0)  # orders/month
feat["return_rate"] = feat["total_returns"] / feat["total_orders"]
feat["discount_usage_rate"] = feat["discount_orders"] / feat["total_orders"]

# favourite category (mode) per customer
fav_cat = (tx_hist.groupby(["customer_id", "category"]).size()
           .reset_index(name="n").sort_values("n", ascending=False)
           .drop_duplicates("customer_id")[["customer_id", "category"]]
           .rename(columns={"category": "favorite_category"}))
feat = feat.merge(fav_cat, on="customer_id", how="left")

# reviews -> avg rating, review count (history only)
rev_hist = reviews[reviews.review_date <= FEATURE_CUTOFF]
rev_agg = rev_hist.groupby("customer_id").agg(
    avg_review_rating=("rating", "mean"),
    n_reviews=("rating", "count"),
).reset_index()
feat = feat.merge(rev_agg, on="customer_id", how="left")

# sessions -> activity intensity in last 90 days before cutoff
sess_hist = sessions[sessions.month <= FEATURE_CUTOFF]
sess_recent = sess_hist[sess_hist.month > (FEATURE_CUTOFF - timedelta(days=90))]
sess_agg = sess_recent.groupby("customer_id").agg(
    site_visits_last_90d=("site_visits", "sum"),
    avg_pages_per_visit=("avg_pages_per_visit", "mean"),
).reset_index()
feat = feat.merge(sess_agg, on="customer_id", how="left")

# tenure at cutoff
feat = feat.merge(eligible[["customer_id", "signup_date", "region",
                             "acquisition_channel", "age", "primary_category"]],
                   on="customer_id", how="right")
feat["customer_tenure_days"] = (FEATURE_CUTOFF - feat["signup_date"]).dt.days

# Customers with zero orders in history: fill sensible defaults (never purchased)
never_purchased = feat["total_orders"].isna()
feat["total_orders"] = feat["total_orders"].fillna(0)
feat["total_spend"] = feat["total_spend"].fillna(0)
feat["average_order_value"] = feat["average_order_value"].fillna(0)
feat["days_since_last_purchase"] = feat["days_since_last_purchase"].fillna(feat["customer_tenure_days"])
feat["purchase_frequency"] = feat["purchase_frequency"].fillna(0)
feat["return_rate"] = feat["return_rate"].fillna(0)
feat["discount_usage_rate"] = feat["discount_usage_rate"].fillna(0)
feat["distinct_categories"] = feat["distinct_categories"].fillna(0)
feat["avg_review_rating"] = feat["avg_review_rating"].fillna(feat["avg_review_rating"].median())
feat["n_reviews"] = feat["n_reviews"].fillna(0)
feat["site_visits_last_90d"] = feat["site_visits_last_90d"].fillna(0)
feat["avg_pages_per_visit"] = feat["avg_pages_per_visit"].fillna(0)
feat["favorite_category"] = feat["favorite_category"].fillna(feat["primary_category"])

# ---------------------------------------------------------------------------
# Customer Lifetime Value (simple historical CLV proxy: total spend so far,
# annualised by tenure - a standard, explainable CLV proxy for this exercise)
# ---------------------------------------------------------------------------
feat["customer_lifetime_value"] = np.where(
    feat["customer_tenure_days"] > 0,
    feat["total_spend"] / feat["customer_tenure_days"] * 365,
    0
)

# ---------------------------------------------------------------------------
# Label: did the customer purchase again in the 60-day window after cutoff?
# churned = 1 if NO purchase in that window (i.e. will NOT purchase again soon)
# ---------------------------------------------------------------------------
purchased_in_window = set(tx_future.customer_id.unique())
feat["will_purchase_next_60d"] = feat["customer_id"].isin(purchased_in_window).astype(int)
feat["churned"] = 1 - feat["will_purchase_next_60d"]

# Drop customers with zero prior history (nothing to predict from / not yet real users)
feat = feat[feat["total_orders"] > 0].reset_index(drop=True)

feat.to_csv(f"{OUT}/customer_features.csv", index=False)

print("Feature table shape:", feat.shape)
print("\nChurn rate (60-day forward-looking):", feat.churned.mean().round(3))
print("\nColumns:", list(feat.columns))
print("\nSample:")
print(feat.head(3).T)
