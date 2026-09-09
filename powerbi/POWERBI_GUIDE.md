# Power BI Dashboard — Build Guide

A `.pbix` can't be authored outside Power BI Desktop itself, so this guide gives the
exact steps to build the 4-page dashboard from the CSVs in this folder. Import all
files via **Get Data → Text/CSV**, then follow each page below.

## Data model
Load all 11 CSVs. Set relationships (Model view):
- `customer_analysis[customer_id]` → `ml_predictions[customer_id]` (1:1)
- `customer_analysis[customer_id]` → `retention_recommendations[customer_id]` (1:1)
- `monthly_business_overview[order_month]` used standalone as the date table for Page 1

## Page 1 — Business Overview
Source: `kpi_cards.csv`, `monthly_business_overview.csv`
- 5 **Card** visuals: Total Revenue, Total Orders, Active Customers, Repeat-Purchase
  Rate (%), Churn Rate (%) — from `kpi_cards.csv`
- **Line chart**: `order_month` (x) vs `revenue` and `orders` (y, dual axis) —
  from `monthly_business_overview.csv`
- **Line chart**: `order_month` vs `repeat_purchase_rate` and `return_rate`

## Page 2 — Customer Analysis
Source: `customer_analysis.csv`, `segment_summary.csv`
- **Donut chart**: customers by `segment`
- **Bar chart**: `avg_clv` by `segment` (from `segment_summary.csv`)
- **Scatter chart**: `purchase_frequency` (x) vs `customer_lifetime_value` (y),
  color = `segment`, size = `total_spend`
- **Line chart**: retention trend — bin `customer_tenure_days` into buckets
  (e.g. 0–90, 90–180, 180–365, 365+) and plot average `churn_probability` per bucket
- Slicers: `region`, `acquisition_channel`, `favorite_category`

## Page 3 — ML Predictions
Source: `ml_predictions.csv`, `top_churn_drivers.csv`, `model_comparison.csv`, `high_risk_customers.csv`
- **Table**: `model_comparison.csv` — accuracy / precision / recall / F1 / ROC-AUC / PR-AUC per model
- **Bar chart**: `top_churn_drivers.csv` — `driver` (y) vs `mean_abs_shap_impact` (x), sorted descending
- **Histogram**: distribution of `churn_probability` (from `ml_predictions.csv`) — use
  a calculated column to bin into deciles
- **Table**: `high_risk_customers.csv` sorted by `customer_lifetime_value` descending —
  this is the actionable "who to call first" list

## Page 4 — Recommendations
Source: `retention_recommendations.csv`, `strategy_comparison.csv`, `action_summary.csv`
- **Table**: `strategy_comparison.csv` — LP-optimized vs Naive baseline, with a
  calculated column `% improvement = (expected_clv_saved - naive) / naive`
- **Stacked bar**: `action_summary.csv` — `assigned_action` (x) vs `customers` and
  `total_cost` — shows how the budget is split across email / discount / personal outreach
- **Card**: headline number — "Expected CLV saved: ₹32.3L vs ₹19.4L naive (+66.7%)"
- **Table**: `retention_recommendations.csv` filtered to `assigned_action != "none"`,
  sorted by `expected_value_saved` descending — the actual call list for the retention team

## Suggested DAX measures
```
Churn Rate = AVERAGE(ml_predictions[predicted_churn])
Repeat Purchase Rate = AVERAGE(monthly_business_overview[repeat_purchase_rate])
Expected CLV Saved (LP) = SUM(retention_recommendations[expected_value_saved])
ROI per Rupee = DIVIDE([Expected CLV Saved (LP)], SUM(retention_recommendations[assigned_cost]))
```
