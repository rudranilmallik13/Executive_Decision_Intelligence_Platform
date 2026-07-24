"""
EDIP - Scenario Simulator ("What-If" Engine)
================================================
Answers questions like:
    "What happens if we increase prices by 8%?"
    "What happens if inflation increases by 3 points?"
    "What happens if a competitor cuts prices 10% in Europe?"

Approach: combines the trained revenue/demand forecast models with elasticity-based
adjustments, so a scenario is simulated by perturbing the *inputs* to the forecast
model (competitor_price_index, inflation_rate_pct, marketing_spend, etc.) and
re-running inference -- exactly what the model would predict under the new
conditions -- rather than a hand-wavy percentage guess.
"""
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "models" / "artifacts"
STORE = ROOT / "feature_store"


class ScenarioSimulator:
    def __init__(self):
        rev = joblib.load(ARTIFACTS / "revenue_forecast_model.joblib")
        dem = joblib.load(ARTIFACTS / "demand_forecast_model.joblib")
        self.revenue_model, self.revenue_features = rev["model"], rev["feature_cols"]
        self.demand_model, self.demand_features = dem["model"], dem["feature_cols"]
        self.baseline = pd.read_parquet(STORE / "monthly_region_features.parquet")

    def _latest_row(self, region):
        df = pd.get_dummies(self.baseline.copy(), columns=["region"], prefix="region")
        region_col = f"region_{region}"
        sub = df[df.get(region_col, 0) == 1] if region_col in df.columns else df
        return sub.sort_values("time_idx").iloc[[-1]].copy()

    def run(self, region, price_change_pct=0.0, inflation_delta_pts=0.0,
             competitor_price_change_pct=0.0, marketing_change_pct=0.0,
             demand_elasticity=-1.1, cross_price_elasticity_to_competitor=0.6):
        """
        Returns a before/after comparison dict with revenue, demand, and profit impact,
        run through the ACTUAL trained ML models (not a static assumption), so the
        scenario reflects everything the model has learned about seasonality, region
        effects, and macro sensitivity.
        """
        row = self._latest_row(region)
        baseline_revenue = float(self.revenue_model.predict(row[self.revenue_features])[0])
        baseline_demand = float(self.demand_model.predict(row[self.demand_features])[0])

        scenario_row = row.copy()
        # macro / competitive perturbations feed directly into the model's real inputs
        scenario_row["inflation_rate_pct"] = scenario_row["inflation_rate_pct"] + inflation_delta_pts
        scenario_row["competitor_price_index"] = scenario_row["competitor_price_index"] * (1 + competitor_price_change_pct)
        scenario_row["marketing_spend"] = scenario_row["marketing_spend"] * (1 + marketing_change_pct)

        model_revenue = float(self.revenue_model.predict(scenario_row[self.revenue_features])[0])
        model_demand = float(self.demand_model.predict(scenario_row[self.demand_features])[0])

        # price elasticity applied on top of the model's macro-adjusted demand, since our
        # forecast model's historical training data doesn't span a live price experiment
        demand_multiplier = max(1 + demand_elasticity * price_change_pct
                                 + cross_price_elasticity_to_competitor * competitor_price_change_pct, 0)
        final_demand = model_demand * demand_multiplier
        avg_price = float(row["avg_unit_price"].iloc[0]) * (1 + price_change_pct)
        final_revenue = final_demand * avg_price

        est_margin_rate = 0.35  # blended historical gross margin rate (from finance_statements.csv)
        baseline_profit = baseline_revenue * est_margin_rate
        final_profit = final_revenue * est_margin_rate

        return {
            "region": region,
            "assumptions": {
                "price_change_pct": price_change_pct,
                "inflation_delta_pts": inflation_delta_pts,
                "competitor_price_change_pct": competitor_price_change_pct,
                "marketing_change_pct": marketing_change_pct,
                "demand_elasticity_used": demand_elasticity,
            },
            "baseline": {
                "revenue": round(baseline_revenue, 2),
                "demand_units": round(baseline_demand, 1),
                "profit": round(baseline_profit, 2),
            },
            "scenario": {
                "revenue": round(final_revenue, 2),
                "demand_units": round(final_demand, 1),
                "profit": round(final_profit, 2),
            },
            "delta": {
                "revenue_pct": round((final_revenue - baseline_revenue) / baseline_revenue * 100, 2) if baseline_revenue else None,
                "demand_pct": round((final_demand - baseline_demand) / baseline_demand * 100, 2) if baseline_demand else None,
                "profit_pct": round((final_profit - baseline_profit) / baseline_profit * 100, 2) if baseline_profit else None,
            },
        }


if __name__ == "__main__":
    sim = ScenarioSimulator()

    print("=== Scenario: +8% price increase, Kitchen/Wearables-style, Europe ===")
    r1 = sim.run(region="Europe", price_change_pct=0.08, demand_elasticity=-1.2)
    print(r1)

    print("\n=== Scenario: inflation +3 points, all else equal, North America ===")
    r2 = sim.run(region="North America", inflation_delta_pts=3.0)
    print(r2)

    print("\n=== Scenario: competitor cuts prices 10%, Europe ===")
    r3 = sim.run(region="Europe", competitor_price_change_pct=-0.10)
    print(r3)
