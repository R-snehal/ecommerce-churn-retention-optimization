# E-Commerce Customer Churn Prediction & Constrained Retention Budget Optimization

**Question answered:** Which customers are likely to stop purchasing, why, and what
should the business do about it — under a fixed retention budget?

---

## 1. Data

Synthetic-but-causally-structured dataset (documented in `src/01_generate_data.py`):
6,000 customers, 76,211 transactions (2023–2025), 41,798 reviews, 94,101 monthly
session records, across 8 categories, 5 regions, 6 acquisition channels. Used instead
of a single public source because no public dataset combines purchase history,
returns, discounts, reviews, *and* session activity at customer level. The data was
generated with an explicit, non-trivial data-generating process (a latent
"engagement" persona plus randomised "churn shocks" that decay purchase rate,
raise return probability, and depress review scores) — churn is not a hand-coded
function of one feature, so the model has to discover the pattern.

**Churn definition:** a customer is labelled *churned* if they made no purchase in
the 60 days after a feature cutoff (2025-07-02); all features are computed strictly
before that cutoff to avoid label leakage. This mirrors a genuine 60-day
forward-looking prediction task. Base rate: **35.2% of customers churn** in the
next 60 days.

## 2. EDA — answers to the five business questions

| Question | Finding |
|---|---|
| Who purchases most frequently? | Most customers order 0.8–1.6×/month (median 1.1); a small high-frequency tail exists but is not where most CLV sits — CLV is driven more by consistency than raw frequency. |
| What causes inactivity? | Active customers averaged **41 days** since last purchase vs. **88 days** for churned customers; active customers had ~19 site visits in the last 90 days vs. ~12 for churned. Review scores were also lower for churners (4.05 vs 3.83 avg rating). |
| Does discount usage affect retention? | Almost no relationship — churn rate is ~34–38% across every discount-usage band. Discounting is **not** a reliable retention lever on its own in this data (correlation with churn: 0.002). |
| Which categories repeat best? | Fashion (57.1% repeat rate) and Electronics (50.8%) lead; Books (43.1%) and Toys & Games (43.3%) lag. |
| How does frequency trend over time? | Monthly orders and unique buyers both grew steadily from ~4,070 orders/2,650 buyers (Mar 2025) to ~4,260 orders/2,800 buyers (Jul 2025), a healthy top-of-funnel trend even as ~35% of the base is individually at churn risk. |

Charts: `outputs/eda/01`–`06_*.png`. Full numeric summary: `outputs/eda/eda_summary.txt`.

## 3. Feature engineering

Built from raw transactions/reviews/sessions to customer level: `total_orders`,
`average_order_value`, `days_since_last_purchase`, `purchase_frequency`,
`return_rate`, `discount_usage_rate`, `customer_lifetime_value` (annualised spend
rate), `customer_tenure_days`, `avg_review_rating`, `site_visits_last_90d`, plus
categorical context (favorite category, region, acquisition channel).

## 4. Model comparison

Trained on a stratified 75/25 split, evaluated on held-out data:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| **Random Forest** | **0.723** | 0.597 | 0.648 | 0.621 | **0.771** | **0.662** |
| Logistic Regression | 0.691 | 0.547 | 0.698 | 0.613 | 0.763 | 0.637 |
| XGBoost | 0.696 | 0.560 | 0.626 | 0.591 | 0.756 | 0.635 |
| Decision Tree | 0.647 | 0.498 | 0.652 | 0.565 | 0.728 | 0.592 |

**Random Forest** is the best model on ROC-AUC and PR-AUC and was used for scoring
and SHAP explanation. Note these are realistic AUCs (0.73–0.77), not inflated —
churn from noisy, multi-cause behavioural data is a genuinely hard problem; a model
claiming >0.95 AUC on this kind of task should be treated with suspicion (usually
label leakage).

## 5. Why — SHAP-based explanation

Ranked by mean |SHAP value| (impact on predicted churn probability):

1. **`site_visits_last_90d`** (0.094) — by far the strongest signal. Falling site
   engagement precedes purchase inactivity, and is observable *before* a customer
   fully stops buying — this is the earliest warning sign available.
2. **`days_since_last_purchase`** (0.045) — the classic recency signal, second
   strongest.
3. **`avg_review_rating`** (0.041) — dissatisfaction (reflected in review scores)
   is a real churn driver, not just a symptom.
4. **`customer_lifetime_value`** (0.028) and **`purchase_frequency`** (0.027) —
   lower-value, less-frequent customers churn more, as expected.
5. **`customer_tenure_days`** (0.017) — longer-tenured customers churn *slightly*
   more in this data, likely reflecting natural lifecycle decay rather than new-user
   risk — worth validating against real cohort data.

Notably, **`discount_usage_rate` ranks 17th of 18** (0.003) — consistent with the
EDA finding that discounting doesn't meaningfully predict retention. This is an
actionable negative finding: broad discounting is a weak retention lever; site
engagement and satisfaction are the real levers.

Charts: `outputs/models/shap_summary_beeswarm.png`, `shap_top_drivers_bar.png`.

## 6. From prediction to decision: constrained retention budget optimization

Most churn projects stop at a risk score and a bullet-point segment list. Here the
churn probabilities are fed into a **linear program** (`src/06_budget_optimization.py`,
solved with PuLP/CBC) that decides, for each of the 3,434 customers at ≥35% churn
risk, whether to send nothing, an email nudge (₹20, 5% churn-reduction), a discount
offer (₹250, 18% reduction), or personal outreach (₹900, 35% reduction, capacity-
capped at 150 calls/cycle) — maximizing **expected CLV saved** subject to a
**₹500,000 budget** and the outreach capacity constraint.

**Effectiveness assumptions are explicit placeholders** (no historical A/B test
exists yet); they should be replaced with real uplift-test results before production
use — the LP structure itself doesn't change.

| Strategy | Customers targeted | Spend | Expected CLV saved | ROI per ₹ spent |
|---|---|---|---|---|
| **LP-optimized (constrained)** | 3,266 | ₹500,000 | **₹3,230,980** | **6.46** |
| Naive top-N riskiest (discount only) | 2,000 | ₹500,000 | ₹1,938,743 | 3.88 |

**The LP allocation beats naive top-N-riskiest targeting by 66.7% in expected CLV
saved, for the same budget.** The gain comes from *not* spending ₹250 discounts on
every risky customer — many are better served by a ₹20 email nudge (spreading reach
across 1,800 people) while the 150 highest-value at-risk customers get the ₹900
personal-outreach treatment the LP can afford precisely because it saved money
elsewhere.

Optimal allocation: 1,800 → email nudge, 1,316 → discount offer, 150 → personal
outreach (capacity-capped), 168 → no action (expected value too low to justify any
cost). Full allocation: `outputs/optimization/lp_retention_allocation.csv`.

## 7. Business recommendations

1. **Fund the LP-based allocation, not a flat discount campaign.** At the current
   ₹500K/cycle budget, this is a ₹1.29M higher expected CLV-saved outcome for the
   same spend — re-run the LP each cycle with fresh churn scores and a real budget
   figure once available.
2. **Treat `site_visits_last_90d` as an early-warning KPI**, not just
   `days_since_last_purchase`. Engagement drops ahead of purchase drops — retention
   triggers (email, app push) should fire on falling session activity, before the
   customer has technically gone 60+ days without an order.
3. **Investigate satisfaction directly**, not just discount harder. Review rating is
   a top-5 churn driver and discount usage is not — a low-rating customer getting a
   discount is being offered the wrong fix. Route customers with `avg_review_rating`
   < 3 and `return_rate` above category baseline to a support/quality follow-up
   instead of a coupon.
4. **Reserve personal outreach for the capacity-constrained 150 highest-value,
   highest-risk customers per cycle** — the LP already identifies exactly who these
   are (`segment == "High Risk / High Value"` in `powerbi/segment_summary.csv`);
   don't dilute this scarce, expensive channel across the full risk pool.
5. **Re-validate discounting as a lever with a real A/B test.** The current data
   shows no retention benefit from broad discounting; before cutting discount spend
   entirely, run a controlled test to confirm this isn't specific to the synthetic
   generative process, since the LP's economics change materially if discount
   effectiveness turns out to be higher (or lower) than assumed here.

## 8. Resume bullet

> Built a customer churn prediction model (Random Forest, ROC-AUC 0.77) and a
> constrained linear-programming retention budget allocator (PuLP) that
> outperformed naive top-N-riskiest targeting by **66.7% in expected customer
> lifetime value saved**, under an identical ₹500K retention budget — turning
> churn scores into an actual resource-allocation decision, not just a risk list.

## Files
```
churn_project/
├── data/
│   ├── raw/                       customers, transactions, reviews, sessions
│   └── processed/                 customer_features.csv, customer_features_scored.csv
├── src/                           01_generate_data.py ... 07_powerbi_exports.py (run in order)
├── outputs/
│   ├── eda/                       6 charts + eda_summary.txt
│   ├── models/                    model_comparison.csv, ROC curves, SHAP plots, best_model.pkl
│   └── optimization/              LP allocation, naive baseline, strategy comparison
├── powerbi/                       11 CSVs + POWERBI_GUIDE.md (4-page dashboard build steps)
└── report/                        this file
```
