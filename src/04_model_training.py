"""
04_model_training.py
---------------------
Trains Logistic Regression, Decision Tree, Random Forest, XGBoost on the
60-day forward-looking churn label. Compares them on held-out test data
(accuracy, precision, recall, F1, ROC-AUC, PR-AUC) and saves:
  - outputs/models/model_comparison.csv
  - outputs/models/roc_curves.png
  - outputs/models/best_model.pkl  (+ preprocessing pipeline)
  - data/processed/customer_features_scored.csv (with predicted churn probability for ALL customers)
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib, os, json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              roc_auc_score, average_precision_score, roc_curve, confusion_matrix)
from xgboost import XGBClassifier

OUT = "/home/claude/churn_project/outputs/models"
os.makedirs(OUT, exist_ok=True)

feat = pd.read_csv("/home/claude/churn_project/data/processed/customer_features.csv")

NUMERIC = ["total_orders","average_order_value","days_since_last_purchase","purchase_frequency",
           "return_rate","discount_usage_rate","distinct_categories","avg_review_rating",
           "n_reviews","site_visits_last_90d","avg_pages_per_visit","customer_tenure_days",
           "customer_lifetime_value","age","total_spend"]
CATEGORICAL = ["favorite_category","region","acquisition_channel"]
TARGET = "churned"

X = feat[NUMERIC + CATEGORICAL].copy()
y = feat[TARGET].copy()
cust_ids = feat["customer_id"].copy()

X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
    X, y, cust_ids, test_size=0.25, random_state=42, stratify=y
)

preprocess = ColumnTransformer([
    ("num", StandardScaler(), NUMERIC),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
])

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=30, random_state=42,
                                             class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=400, max_depth=10, min_samples_leaf=10,
                                             random_state=42, class_weight="balanced", n_jobs=-1),
    "XGBoost": XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                              random_state=42,
                              scale_pos_weight=(y_train==0).sum()/(y_train==1).sum()),
}

results = []
fitted = {}
plt.figure(figsize=(7,6))

for name, clf in models.items():
    pipe = Pipeline([("prep", preprocess), ("clf", clf)])
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    row = {
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
        "pr_auc": average_precision_score(y_test, proba),
    }
    results.append(row)
    fitted[name] = pipe

    fpr, tpr, _ = roc_curve(y_test, proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC={row['roc_auc']:.3f})")

plt.plot([0,1],[0,1],"k--",alpha=0.4)
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curves - Churn Prediction Models")
plt.legend(); plt.tight_layout()
plt.savefig(f"{OUT}/roc_curves.png", dpi=130); plt.close()

results_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
results_df.to_csv(f"{OUT}/model_comparison.csv", index=False)
print(results_df.round(3).to_string(index=False))

best_name = results_df.iloc[0]["model"]
best_pipe = fitted[best_name]
print(f"\nBest model by ROC-AUC: {best_name}")

joblib.dump(best_pipe, f"{OUT}/best_model.pkl")
joblib.dump(fitted, f"{OUT}/all_models.pkl")
with open(f"{OUT}/best_model_name.json","w") as f:
    json.dump({"best_model": best_name}, f)

# Confusion matrix for best model
pred_best = (best_pipe.predict_proba(X_test)[:,1] >= 0.5).astype(int)
cm = confusion_matrix(y_test, pred_best)
print(f"\nConfusion matrix ({best_name}):\n", cm)

# Score ALL customers (train+test) with the best model for downstream use
all_proba = best_pipe.predict_proba(X)[:, 1]
feat_scored = feat.copy()
feat_scored["churn_probability"] = all_proba
feat_scored["predicted_churn"] = (all_proba >= 0.5).astype(int)
feat_scored["is_test_set"] = feat_scored["customer_id"].isin(set(id_test)).astype(int)
feat_scored.to_csv("/home/claude/churn_project/data/processed/customer_features_scored.csv", index=False)

print("\nSaved: model_comparison.csv, roc_curves.png, best_model.pkl, customer_features_scored.csv")
