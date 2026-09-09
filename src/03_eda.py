"""
03_eda.py
---------
Exploratory analysis answering:
  1. Which customers purchase most frequently?
  2. What causes customers to become inactive (churn)?
  3. Does discount usage affect retention?
  4. Which product categories have the highest repeat purchases?
  5. How does purchase frequency change over time (cohort trend)?
Saves charts to outputs/eda/*.png and a text summary to outputs/eda/eda_summary.txt
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme(style="whitegrid", palette="deep")
OUT = "/home/claude/churn_project/outputs/eda"
os.makedirs(OUT, exist_ok=True)

feat = pd.read_csv("/home/claude/churn_project/data/processed/customer_features.csv",
                    parse_dates=["signup_date", "first_purchase", "last_purchase"])
tx = pd.read_csv("/home/claude/churn_project/data/raw/transactions.csv", parse_dates=["order_date"])

summary_lines = []
def log(s):
    print(s)
    summary_lines.append(str(s))

log("="*70)
log("EDA SUMMARY")
log("="*70)

# 1. Purchase frequency distribution & top-frequency customers -------------
log(f"\n1) PURCHASE FREQUENCY")
log(feat["purchase_frequency"].describe().round(2).to_string())
top_freq = feat.nlargest(10, "purchase_frequency")[["customer_id","purchase_frequency","total_orders","customer_lifetime_value"]]
log("\nTop 10 most frequent purchasers:\n" + top_freq.to_string(index=False))

plt.figure(figsize=(8,5))
sns.histplot(feat["purchase_frequency"].clip(upper=feat["purchase_frequency"].quantile(0.99)), bins=40, kde=True)
plt.title("Distribution of Purchase Frequency (orders/month)")
plt.xlabel("Orders per month"); plt.tight_layout()
plt.savefig(f"{OUT}/01_purchase_frequency_dist.png", dpi=130); plt.close()

# 2. Drivers of churn (inactivity) ------------------------------------------
log(f"\n2) CHURN BY KEY DRIVER (mean feature value: active vs churned)")
drivers = ["days_since_last_purchase","purchase_frequency","return_rate","avg_review_rating",
           "discount_usage_rate","customer_tenure_days","site_visits_last_90d","customer_lifetime_value"]
comp = feat.groupby("churned")[drivers].mean().T
comp.columns = ["active(0)","churned(1)"]
log(comp.round(2).to_string())

fig, axes = plt.subplots(2, 2, figsize=(12,9))
sns.boxplot(data=feat, x="churned", y="days_since_last_purchase", ax=axes[0,0]); axes[0,0].set_title("Days Since Last Purchase")
sns.boxplot(data=feat, x="churned", y="return_rate", ax=axes[0,1]); axes[0,1].set_title("Return Rate")
sns.boxplot(data=feat, x="churned", y="avg_review_rating", ax=axes[1,0]); axes[1,0].set_title("Avg Review Rating")
sns.boxplot(data=feat, x="churned", y="site_visits_last_90d", ax=axes[1,1]); axes[1,1].set_title("Site Visits (last 90d)")
for ax in axes.flat: ax.set_xlabel("Churned (0=Active, 1=Churned)")
plt.tight_layout(); plt.savefig(f"{OUT}/02_churn_drivers_boxplots.png", dpi=130); plt.close()

# 3. Discount usage vs retention --------------------------------------------
feat["discount_band"] = pd.cut(feat["discount_usage_rate"], [-0.01,0,0.25,0.5,0.75,1.0],
                                 labels=["0%","1-25%","26-50%","51-75%","76-100%"])
disc_churn = feat.groupby("discount_band", observed=True)["churned"].mean()
log(f"\n3) CHURN RATE BY DISCOUNT-USAGE BAND\n{disc_churn.round(3).to_string()}")

plt.figure(figsize=(7,5))
disc_churn.plot(kind="bar", color=sns.color_palette("deep")[3])
plt.ylabel("Churn rate"); plt.title("Churn Rate by Share of Orders Using a Discount")
plt.xticks(rotation=0); plt.tight_layout()
plt.savefig(f"{OUT}/03_discount_vs_churn.png", dpi=130); plt.close()

# 4. Repeat purchase rate by category ----------------------------------------
cat_orders = tx.groupby(["customer_id","category"]).size().reset_index(name="orders_in_cat")
cat_repeat = cat_orders.groupby("category").apply(
    lambda d: (d["orders_in_cat"] > 1).mean(), include_groups=False
).sort_values(ascending=False)
log(f"\n4) REPEAT-PURCHASE RATE BY CATEGORY (share of customers in category with >1 order)\n{cat_repeat.round(3).to_string()}")

plt.figure(figsize=(9,5))
cat_repeat.plot(kind="bar", color=sns.color_palette("deep")[0])
plt.ylabel("Repeat purchase rate"); plt.title("Repeat-Purchase Rate by Product Category")
plt.xticks(rotation=35, ha="right"); plt.tight_layout()
plt.savefig(f"{OUT}/04_repeat_purchase_by_category.png", dpi=130); plt.close()

# 5. Purchase frequency trend over time (monthly order volume + unique buyers)
tx["order_month"] = tx["order_date"].values.astype("datetime64[M]")
monthly = tx.groupby("order_month").agg(orders=("order_id","count"),
                                          unique_buyers=("customer_id","nunique"),
                                          revenue=("net_amount","sum")).reset_index()
log(f"\n5) MONTHLY TREND (last 6 months)\n{monthly.tail(6).round(0).to_string(index=False)}")

fig, ax1 = plt.subplots(figsize=(11,5))
ax1.plot(monthly.order_month, monthly.orders, marker="o", label="Orders")
ax1.plot(monthly.order_month, monthly.unique_buyers, marker="s", label="Unique buyers")
ax1.set_ylabel("Count"); ax1.legend(loc="upper left")
plt.title("Monthly Orders & Unique Buyers Over Time")
plt.xticks(rotation=45); plt.tight_layout()
plt.savefig(f"{OUT}/05_monthly_trend.png", dpi=130); plt.close()

# Correlation heatmap of numeric features vs churn
num_cols = ["churned","days_since_last_purchase","total_orders","average_order_value",
            "purchase_frequency","return_rate","discount_usage_rate","avg_review_rating",
            "customer_tenure_days","customer_lifetime_value","site_visits_last_90d"]
corr = feat[num_cols].corr()
plt.figure(figsize=(9,7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Matrix (numeric features vs churn)")
plt.tight_layout(); plt.savefig(f"{OUT}/06_correlation_heatmap.png", dpi=130); plt.close()
log(f"\n6) CORRELATION WITH CHURN (sorted)\n{corr['churned'].drop('churned').sort_values(key=abs, ascending=False).round(3).to_string()}")

with open(f"{OUT}/eda_summary.txt","w") as f:
    f.write("\n".join(summary_lines))

print("\nSaved charts + summary to", OUT)
