"""
06_budget_optimization.py
--------------------------
Turns churn scores into a DECISION: given a fixed retention budget, which
customers should get which retention action, to maximize expected CLV saved?

Setup
-----
For each customer i, we can offer one of several retention actions a in {none, email,
discount, personal_outreach}, each with:
  - a cost c_a (INR)
  - an assumed effectiveness e_a = reduction in churn probability (uplift), a
    standard assumption used when no historical A/B test exists yet (stated
    explicitly as an assumption customers can calibrate later with real test data)
  - a capacity limit for the most expensive action (agents can only call so many
    people): personal_outreach capacity constraint

Expected value saved by acting on customer i with action a:
    value_i,a = churn_probability_i * effectiveness_a * customer_lifetime_value_i

We only intervene on customers already predicted at meaningful churn risk
(probability > 0.35) - spending budget on already-loyal customers wastes it.

Decision variables: x[i,a] in {0,1} - assign customer i to action a (or none)
Objective: maximize sum(value_i,a * x[i,a])
Constraints:
  - each customer gets at most 1 action
  - sum(cost_a * x[i,a]) <= BUDGET
  - sum(x[i,'personal_outreach']) <= OUTREACH_CAPACITY

We solve this with PuLP (CBC solver) and compare the result against a naive
baseline: "just target the top-N riskiest customers with the discount action
until the budget runs out" - the kind of heuristic most churn projects stop at.
"""
import pandas as pd
import numpy as np
import pulp
import os

OUT = "/home/claude/churn_project/outputs/optimization"
os.makedirs(OUT, exist_ok=True)

feat = pd.read_csv("/home/claude/churn_project/data/processed/customer_features_scored.csv")

# ---- business assumptions (documented, easy to swap for real test data) ----
BUDGET = 500_000          # INR, one retention campaign cycle
OUTREACH_CAPACITY = 150   # personal calls the retention team can realistically make
RISK_THRESHOLD = 0.35     # only consider customers at/above this predicted churn probability

ACTIONS = {
    "none":              {"cost": 0,    "effectiveness": 0.00},
    "email_nudge":       {"cost": 20,   "effectiveness": 0.05},   # cheap, low uplift
    "discount_offer":    {"cost": 250,  "effectiveness": 0.18},   # moderate cost, moderate uplift
    "personal_outreach": {"cost": 900,  "effectiveness": 0.35},   # expensive, high uplift, capacity-limited
}

candidates = feat[feat["churn_probability"] >= RISK_THRESHOLD].copy().reset_index(drop=True)
print(f"Candidates at/above risk threshold ({RISK_THRESHOLD}): {len(candidates)} of {len(feat)} customers")

# Expected value saved per customer per action
for a, p in ACTIONS.items():
    candidates[f"value_{a}"] = candidates["churn_probability"] * p["effectiveness"] * candidates["customer_lifetime_value"]

# ---------------------------------------------------------------------------
# LP formulation
# ---------------------------------------------------------------------------
prob = pulp.LpProblem("Retention_Budget_Allocation", pulp.LpMaximize)

action_list = list(ACTIONS.keys())
idx = candidates.index.tolist()

x = pulp.LpVariable.dicts("assign", (idx, action_list), cat="Binary")

# Objective: maximize expected CLV saved
prob += pulp.lpSum(
    candidates.loc[i, f"value_{a}"] * x[i][a] for i in idx for a in action_list
)

# Each customer gets exactly one action (including "none")
for i in idx:
    prob += pulp.lpSum(x[i][a] for a in action_list) == 1

# Budget constraint
prob += pulp.lpSum(ACTIONS[a]["cost"] * x[i][a] for i in idx for a in action_list) <= BUDGET

# Capacity constraint on the expensive, high-touch action
prob += pulp.lpSum(x[i]["personal_outreach"] for i in idx) <= OUTREACH_CAPACITY

solver = pulp.PULP_CBC_CMD(msg=0)
prob.solve(solver)

print("LP status:", pulp.LpStatus[prob.status])

# Extract solution
assignment = []
for i in idx:
    for a in action_list:
        if x[i][a].value() == 1:
            assignment.append(a)
            break
candidates["assigned_action"] = assignment
candidates["assigned_cost"] = candidates["assigned_action"].map(lambda a: ACTIONS[a]["cost"])
candidates["expected_value_saved"] = candidates.apply(
    lambda r: r[f"value_{r['assigned_action']}"], axis=1
)

lp_total_value = candidates["expected_value_saved"].sum()
lp_total_cost = candidates["assigned_cost"].sum()

print(f"\nLP OPTIMAL ALLOCATION")
print(candidates["assigned_action"].value_counts().to_string())
print(f"Total spend: Rs {lp_total_cost:,.0f} / budget Rs {BUDGET:,.0f}")
print(f"Total expected CLV saved: Rs {lp_total_value:,.0f}")

# ---------------------------------------------------------------------------
# Naive baseline: rank by churn probability, spend on discount_offer
# (most common heuristic: "target the top-N riskiest with a discount")
# until budget exhausted
# ---------------------------------------------------------------------------
naive = candidates.sort_values("churn_probability", ascending=False).copy()
naive_cost_per = ACTIONS["discount_offer"]["cost"]
max_n = int(BUDGET // naive_cost_per)
naive_selected = naive.iloc[:max_n].copy()
naive_selected["assigned_action"] = "discount_offer"
naive_value = (naive_selected["churn_probability"] * ACTIONS["discount_offer"]["effectiveness"]
               * naive_selected["customer_lifetime_value"]).sum()
naive_cost = len(naive_selected) * naive_cost_per

print(f"\nNAIVE BASELINE (top-N riskiest -> discount_offer only)")
print(f"Customers targeted: {len(naive_selected)}")
print(f"Total spend: Rs {naive_cost:,.0f} / budget Rs {BUDGET:,.0f}")
print(f"Total expected CLV saved: Rs {naive_value:,.0f}")

uplift_pct = (lp_total_value - naive_value) / naive_value * 100
print(f"\n>>> LP allocation beats naive top-N targeting by {uplift_pct:.1f}% in expected CLV saved <<<")

# Save outputs
candidates.to_csv(f"{OUT}/lp_retention_allocation.csv", index=False)
naive_selected.to_csv(f"{OUT}/naive_baseline_allocation.csv", index=False)

comparison = pd.DataFrame([
    {"strategy": "LP-optimized (constrained)", "customers_targeted": (candidates["assigned_action"]!="none").sum(),
     "spend": lp_total_cost, "expected_clv_saved": lp_total_value},
    {"strategy": "Naive top-N riskiest (discount only)", "customers_targeted": len(naive_selected),
     "spend": naive_cost, "expected_clv_saved": naive_value},
])
comparison["roi_per_rupee_spent"] = comparison["expected_clv_saved"] / comparison["spend"]
comparison.to_csv(f"{OUT}/strategy_comparison.csv", index=False)
print("\n", comparison.round(2).to_string(index=False))

print(f"\nSaved LP outputs to {OUT}")
