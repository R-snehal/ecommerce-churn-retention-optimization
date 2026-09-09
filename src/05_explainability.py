"""
05_explainability.py
---------------------
Uses SHAP to explain the best model's predictions:
  - Global feature importance (which factors drive churn overall)
  - Summary/beeswarm plot (direction + magnitude of effect)
  - Saves a ranked driver table used directly in the business writeup
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib, json, os
import shap

OUT = "/home/claude/churn_project/outputs/models"
os.makedirs(OUT, exist_ok=True)

feat = pd.read_csv("/home/claude/churn_project/data/processed/customer_features.csv")
NUMERIC = ["total_orders","average_order_value","days_since_last_purchase","purchase_frequency",
           "return_rate","discount_usage_rate","distinct_categories","avg_review_rating",
           "n_reviews","site_visits_last_90d","avg_pages_per_visit","customer_tenure_days",
           "customer_lifetime_value","age","total_spend"]
CATEGORICAL = ["favorite_category","region","acquisition_channel"]
X = feat[NUMERIC + CATEGORICAL].copy()

with open(f"{OUT}/best_model_name.json") as f:
    best_name = json.load(f)["best_model"]
all_models = joblib.load(f"{OUT}/all_models.pkl")
pipe = all_models[best_name]

prep = pipe.named_steps["prep"]
clf = pipe.named_steps["clf"]

X_trans = prep.transform(X)
feature_names = list(prep.get_feature_names_out())
X_trans_df = pd.DataFrame(X_trans.toarray() if hasattr(X_trans, "toarray") else X_trans,
                           columns=feature_names)

# Use a sample for speed with tree explainer
sample_idx = np.random.RandomState(42).choice(len(X_trans_df), size=min(1500, len(X_trans_df)), replace=False)
X_sample = X_trans_df.iloc[sample_idx]

explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_sample)
# Handle binary classifier output shape variations across sklearn/xgboost versions
if isinstance(shap_values, list):
    sv = shap_values[1]
elif shap_values.ndim == 3:
    sv = shap_values[:, :, 1]
else:
    sv = shap_values

# Global importance table
mean_abs_shap = np.abs(sv).mean(axis=0)
importance_df = pd.DataFrame({
    "feature": feature_names,
    "mean_abs_shap": mean_abs_shap
}).sort_values("mean_abs_shap", ascending=False)

# Collapse one-hot columns back to their original feature for readability
def base_feature(name):
    for c in CATEGORICAL:
        if name.startswith(f"cat__{c}_"):
            return c
    if name.startswith("num__"):
        return name.replace("num__", "")
    return name

importance_df["base_feature"] = importance_df["feature"].apply(base_feature)
collapsed = importance_df.groupby("base_feature")["mean_abs_shap"].sum().sort_values(ascending=False)
collapsed.to_csv(f"{OUT}/shap_feature_importance.csv")
print(f"Model explained: {best_name}")
print("\nTop churn drivers (mean |SHAP value|):")
print(collapsed.head(12).round(4).to_string())

# Beeswarm summary plot
plt.figure()
shap.summary_plot(sv, X_sample, feature_names=feature_names, show=False, max_display=15)
plt.tight_layout()
plt.savefig(f"{OUT}/shap_summary_beeswarm.png", dpi=130, bbox_inches="tight")
plt.close()

# Bar plot of collapsed importance
plt.figure(figsize=(8,6))
collapsed.head(12).sort_values().plot(kind="barh", color="#c0392b")
plt.xlabel("Mean |SHAP value| (impact on churn probability)")
plt.title(f"Top Churn Drivers - {best_name} (SHAP)")
plt.tight_layout()
plt.savefig(f"{OUT}/shap_top_drivers_bar.png", dpi=130)
plt.close()

print("\nSaved SHAP outputs to", OUT)
