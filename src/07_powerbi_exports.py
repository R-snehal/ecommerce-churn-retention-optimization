"""
07_powerbi_exports.py
----------------------
Produces clean, flat CSV tables ready to import into Power BI (Get Data > Text/CSV)
for the 4-page dashboard described in the project brief. Also writes a Power BI
build guide (powerbi/POWERBI_GUIDE.md) with the exact visuals/measures to create,
since a .pbix cannot be authored outside Power BI Desktop itself.
"""
import pandas as pd
import numpy as np
import os

RAW = "/home/claude/churn_project/data/raw"
PROC = "/home/claude/churn_project/data/processed"
OPT = "/home/claude/churn_project/outputs/optimization"
OUT = "/home/claude/churn_project/powerbi"
os.makedirs(OUT, exist_ok=True)

tx = pd.read_csv(f"{RAW}/transactions.csv", parse_dates=["order_date"])
customers = pd.read_csv(f"{RAW}/customers.csv", parse_dates=["signup_date"])
scored = pd.read_csv(f"{PROC}/customer_features_scored.csv")
lp = pd.read_csv(f"{OPT}/lp_retention_allocation.csv")
strategy_comp = pd.read_csv(f"{OPT}/strategy_comparison.csv")

# ---- Page 1: Business Overview (monthly KPIs) ----
tx["order_month"] = tx["order_date"].values.astype("datetime64[M]")
monthly_kpi = tx.groupby("order_month").agg(
    revenue=("net_amount", "sum"),
    orders=("order_id", "count"),
    active_customers=("customer_id", "nunique"),
    returns=("is_returned", "sum"),
).reset_index()
monthly_kpi["return_rate"] = monthly_kpi["returns"] / monthly_kpi["orders"]
# repeat purchase rate per month = share of that month's buyers who had also bought before
first_purchase = tx.groupby("customer_id")["order_date"].min().rename("first_purchase")
tx2 = tx.merge(first_purchase, on="customer_id")
tx2["is_repeat"] = tx2["order_date"] > tx2["first_purchase"]
repeat_by_month = tx2.groupby("order_month")["is_repeat"].mean().rename("repeat_purchase_rate")
monthly_kpi = monthly_kpi.merge(repeat_by_month, on="order_month")
monthly_kpi.to_csv(f"{OUT}/monthly_business_overview.csv", index=False)

overall_churn_rate = scored["churned"].mean()
kpi_cards = pd.DataFrame([{
    "total_revenue": tx["net_amount"].sum(),
    "total_orders": len(tx),
    "active_customers_scored_base": len(scored),
    "overall_repeat_purchase_rate": tx2["is_repeat"].mean(),
    "overall_return_rate": tx["is_returned"].mean(),
    "churn_rate_60d_forward": overall_churn_rate,
}])
kpi_cards.to_csv(f"{OUT}/kpi_cards.csv", index=False)

# ---- Page 2: Customer Analysis ----
def segment(row):
    if row["churn_probability"] >= 0.6 and row["customer_lifetime_value"] >= scored["customer_lifetime_value"].quantile(0.66):
        return "High Risk / High Value"
    elif row["churn_probability"] >= 0.6:
        return "High Risk / Low-Med Value"
    elif row["churn_probability"] < 0.35:
        return "Low Risk (Healthy)"
    else:
        return "Watchlist (Medium Risk)"

scored["segment"] = scored.apply(segment, axis=1)
customer_analysis = scored.merge(
    customers[["customer_id","region","acquisition_channel"]].drop_duplicates(),
    on="customer_id", how="left", suffixes=("","_dup")
)
keep_cols = ["customer_id","segment","churn_probability","predicted_churn","churned",
             "customer_lifetime_value","total_orders","total_spend","average_order_value",
             "purchase_frequency","days_since_last_purchase","return_rate","discount_usage_rate",
             "avg_review_rating","site_visits_last_90d","customer_tenure_days",
             "favorite_category","region","acquisition_channel","age"]
customer_analysis[keep_cols].to_csv(f"{OUT}/customer_analysis.csv", index=False)

segment_summary = scored.groupby("segment").agg(
    customers=("customer_id","count"),
    avg_clv=("customer_lifetime_value","mean"),
    total_clv=("customer_lifetime_value","sum"),
    avg_churn_prob=("churn_probability","mean"),
    actual_churn_rate=("churned","mean"),
).reset_index().sort_values("total_clv", ascending=False)
segment_summary.to_csv(f"{OUT}/segment_summary.csv", index=False)

# ---- Page 3: ML Predictions ----
predictions_export = scored[["customer_id","churn_probability","predicted_churn","churned",
                               "customer_lifetime_value","segment"]].copy()
predictions_export.to_csv(f"{OUT}/ml_predictions.csv", index=False)

shap_imp = pd.read_csv("/home/claude/churn_project/outputs/models/shap_feature_importance.csv")
shap_imp.columns = ["driver","mean_abs_shap_impact"]
shap_imp.to_csv(f"{OUT}/top_churn_drivers.csv", index=False)

model_comp = pd.read_csv("/home/claude/churn_project/outputs/models/model_comparison.csv")
model_comp.to_csv(f"{OUT}/model_comparison.csv", index=False)

high_risk_customers = scored[scored["churn_probability"] >= 0.6].sort_values(
    "customer_lifetime_value", ascending=False
)[["customer_id","churn_probability","customer_lifetime_value","days_since_last_purchase",
   "return_rate","avg_review_rating","favorite_category"]]
high_risk_customers.to_csv(f"{OUT}/high_risk_customers.csv", index=False)

# ---- Page 4: Recommendations (LP allocation + comparison) ----
lp_export = lp[["customer_id","churn_probability","customer_lifetime_value",
                 "assigned_action","assigned_cost","expected_value_saved"]]
lp_export.to_csv(f"{OUT}/retention_recommendations.csv", index=False)
strategy_comp.to_csv(f"{OUT}/strategy_comparison.csv", index=False)

action_summary = lp.groupby("assigned_action").agg(
    customers=("customer_id","count"),
    total_cost=("assigned_cost","sum"),
    total_expected_value_saved=("expected_value_saved","sum"),
).reset_index()
action_summary.to_csv(f"{OUT}/action_summary.csv", index=False)

print("Power BI export files written to", OUT)
for f in sorted(os.listdir(OUT)):
    print(" -", f)
