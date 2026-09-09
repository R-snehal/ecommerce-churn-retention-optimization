"""
01_generate_data.py
--------------------
Generates a realistic, causally-structured synthetic e-commerce dataset.

WHY SYNTHETIC:
No single public dataset contains customer_id + purchase history + order frequency +
AOV + categories + reviews + returns + discounts + session data + tenure together
(UCI Online Retail has transactions only; Olist lacks session/discount data; Kaggle
"ecommerce customer" sets lack raw transactions). To demonstrate the full pipeline
(EDA -> features -> churn model -> SHAP -> budget optimization) end-to-end, we
generate data with an explicit, documented data-generating process (DGP) so that:
  1. Every field the brief asks for is present and consistent with the others.
  2. The churn label is NOT a deterministic function of one feature - it depends on
     multiple correlated, noisy drivers (recency, frequency, returns, satisfaction,
     support tickets, tenure, category), the way real churn does.
  3. Feature importances discovered by the model are a genuine result of the pipeline,
     not something we hard-coded - we do not tell the model the DGP weights.

Output: data/raw/transactions.csv, data/raw/customers.csv
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

rng = np.random.default_rng(42)

N_CUSTOMERS = 6000
OBS_START = datetime(2023, 1, 1)
OBS_END = datetime(2025, 8, 31)          # "today" for recency calcs
TOTAL_DAYS = (OBS_END - OBS_START).days

CATEGORIES = ["Electronics", "Fashion", "Home & Kitchen", "Beauty & Personal Care",
              "Books", "Sports & Outdoors", "Grocery", "Toys & Games"]
CATEGORY_BASE_PRICE = {
    "Electronics": 3200, "Fashion": 900, "Home & Kitchen": 1400,
    "Beauty & Personal Care": 550, "Books": 350, "Sports & Outdoors": 1100,
    "Grocery": 450, "Toys & Games": 700,
}
CATEGORY_RETURN_RATE = {   # baseline return propensity by category (fashion/electronics higher)
    "Electronics": 0.10, "Fashion": 0.14, "Home & Kitchen": 0.06,
    "Beauty & Personal Care": 0.04, "Books": 0.02, "Sports & Outdoors": 0.05,
    "Grocery": 0.01, "Toys & Games": 0.05,
}
REGIONS = ["North", "South", "East", "West", "Central"]
ACQUISITION_CHANNELS = ["Organic Search", "Paid Ads", "Social Media", "Referral", "Email", "Direct"]

# ---------------------------------------------------------------------------
# 1. Customer master (latent "persona" drives everything downstream)
# ---------------------------------------------------------------------------
customer_id = np.arange(100000, 100000 + N_CUSTOMERS)

# signup date: more customers joined recently (growth), some are long-tenured
tenure_days_at_end = rng.gamma(shape=2.0, scale=260, size=N_CUSTOMERS).astype(int)
tenure_days_at_end = np.clip(tenure_days_at_end, 5, TOTAL_DAYS)
signup_date = np.array([OBS_END - timedelta(days=int(d)) for d in tenure_days_at_end])
signup_date = np.array([max(d, OBS_START) for d in signup_date])

region = rng.choice(REGIONS, size=N_CUSTOMERS)
channel = rng.choice(ACQUISITION_CHANNELS, size=N_CUSTOMERS,
                      p=[0.28, 0.20, 0.15, 0.12, 0.10, 0.15])
age = np.clip(rng.normal(35, 11, N_CUSTOMERS), 18, 75).astype(int)

# Latent "engagement propensity" (unobserved persona) - drives frequency, satisfaction, loyalty
engagement = rng.beta(2.2, 3.0, N_CUSTOMERS)          # 0-1, most customers modestly engaged
price_sensitivity = rng.beta(2.5, 2.5, N_CUSTOMERS)   # 0-1, drives discount-seeking behavior
primary_category = rng.choice(CATEGORIES, size=N_CUSTOMERS,
                               p=[0.16, 0.20, 0.14, 0.13, 0.08, 0.10, 0.11, 0.08])

customers = pd.DataFrame({
    "customer_id": customer_id,
    "signup_date": signup_date,
    "region": region,
    "acquisition_channel": channel,
    "age": age,
    "primary_category": primary_category,
    "_engagement": engagement,             # latent, kept for transaction sim, dropped later
    "_price_sensitivity": price_sensitivity,
})

# ---------------------------------------------------------------------------
# 2. Transaction simulation per customer (Poisson process modulated by engagement,
#    with a churn "event" after which purchase rate collapses)
# ---------------------------------------------------------------------------
rows = []
review_rows = []
session_rows = []

for _, c in customers.iterrows():
    cid = c.customer_id
    start = c.signup_date
    days_active = (OBS_END - start).days
    if days_active <= 0:
        continue

    base_rate = 0.008 + 0.05 * c._engagement          # purchases/day
    # Some customers experience a "churn shock" (bad experience) partway through
    # -> after that point purchase rate decays sharply. This is what the model
    # has to learn to detect from recency/return/review signals, not a label leak.
    had_shock = rng.random() < (0.15 + 0.35 * (1 - c._engagement))
    shock_day = rng.integers(int(days_active * 0.2), max(int(days_active * 0.9), int(days_active*0.2)+1)) if had_shock and days_active > 10 else None

    t = 0
    n_orders_cust = 0
    while t < days_active:
        gap = rng.exponential(1 / base_rate)
        t += gap
        if t >= days_active:
            break
        cur_rate_mult = 1.0
        if shock_day is not None and t > shock_day:
            decay = np.exp(-(t - shock_day) / 60.0)   # decays over ~2 months
            cur_rate_mult = 0.15 + 0.85 * decay
            if rng.random() > cur_rate_mult:
                continue  # skip this purchase - customer is disengaging

        order_date = start + timedelta(days=t)
        cat = rng.choice(CATEGORIES) if rng.random() < 0.35 else c.primary_category
        base_price = CATEGORY_BASE_PRICE[cat]
        price = max(50, rng.normal(base_price, base_price * 0.35))
        qty = rng.choice([1, 1, 1, 2, 2, 3], p=[0.45, 0.2, 0.1, 0.15, 0.07, 0.03])

        # discount usage: more likely for price-sensitive customers, promo seasons
        discount_pct = 0.0
        used_discount = rng.random() < (0.15 + 0.55 * c._price_sensitivity)
        if used_discount:
            discount_pct = rng.choice([5, 10, 15, 20, 25, 30], p=[0.25,0.25,0.2,0.15,0.1,0.05])

        gross_amount = price * qty
        net_amount = gross_amount * (1 - discount_pct / 100)

        # returns: category baseline + higher if customer is on the churn path
        return_prob = CATEGORY_RETURN_RATE[cat]
        if shock_day is not None and t > shock_day:
            return_prob += 0.12
        is_returned = rng.random() < return_prob

        n_orders_cust += 1
        order_id = f"ORD{cid}{n_orders_cust:04d}"
        rows.append((order_id, cid, order_date.date().isoformat(), cat, qty,
                     round(price, 2), discount_pct, round(net_amount, 2), int(is_returned)))

        # review: satisfied customers (high engagement, no shock) rate higher
        if rng.random() < 0.55:
            shock_penalty = 1.6 if (shock_day is not None and t > shock_day) else 0.0
            rating = np.clip(rng.normal(4.2 - shock_penalty + 0.4*(c._engagement-0.5), 0.9), 1, 5)
            review_rows.append((order_id, cid, order_date.date().isoformat(), round(rating, 1)))

    # session/activity log: monthly site visits, roughly tracking purchase engagement
    n_months = max(1, days_active // 30)
    for m in range(n_months):
        month_date = start + timedelta(days=m * 30)
        if month_date > OBS_END:
            break
        mult = 1.0
        if shock_day is not None and (m * 30) > shock_day:
            mult = 0.2 + 0.8 * np.exp(-((m*30) - shock_day) / 60.0)
        visits = max(0, int(rng.poisson(2 + 10 * c._engagement * mult)))
        pages_per_visit = max(1, rng.poisson(4 + 3 * c._engagement))
        session_rows.append((cid, month_date.date().isoformat(), visits, pages_per_visit))

transactions = pd.DataFrame(rows, columns=[
    "order_id", "customer_id", "order_date", "category", "quantity",
    "unit_price", "discount_pct", "net_amount", "is_returned"
])
reviews = pd.DataFrame(review_rows, columns=["order_id", "customer_id", "review_date", "rating"])
sessions = pd.DataFrame(session_rows, columns=["customer_id", "month", "site_visits", "avg_pages_per_visit"])

customers_out = customers.drop(columns=["_engagement", "_price_sensitivity"]).copy()
customers_out["signup_date"] = customers_out["signup_date"].dt.date.astype(str)

import os
os.makedirs("/home/claude/churn_project/data/raw", exist_ok=True)
customers_out.to_csv("/home/claude/churn_project/data/raw/customers.csv", index=False)
transactions.to_csv("/home/claude/churn_project/data/raw/transactions.csv", index=False)
reviews.to_csv("/home/claude/churn_project/data/raw/reviews.csv", index=False)
sessions.to_csv("/home/claude/churn_project/data/raw/sessions.csv", index=False)

print("customers:", customers_out.shape)
print("transactions:", transactions.shape)
print("reviews:", reviews.shape)
print("sessions:", sessions.shape)
print("customers with >=1 order:", transactions.customer_id.nunique())
print("OBS_END used as 'as-of' date for recency:", OBS_END.date())
