"""
EDIP - Optimization Engine
============================
Turns predictions into concrete, numeric decisions rather than vague advice.

1. optimize_inventory()      -> Economic Order Quantity + reorder point per SKU/region,
                                  respecting a warehouse budget constraint (Mixed Integer
                                  Program via PuLP).
2. optimize_price()          -> chooses the profit-maximizing price point for a product
                                  from a candidate grid, given an estimated demand
                                  elasticity curve.
3. optimize_marketing_budget()-> allocates a fixed marketing budget across regions to
                                  maximize predicted incremental revenue (Linear Program).
4. select_supplier()          -> chooses supplier(s) to fulfill required volume at
                                  lowest cost subject to a diversification cap
                                  (no single supplier > 40% of volume) (MIP).

All functions return both the optimal decision AND the numbers behind it, so the
Executive Dashboard can show "why" alongside "what".
"""
import numpy as np
import pandas as pd


def _get_pulp():
    try:
        import pulp
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PuLP is required for optimization functions. Install it with 'python -m pip install pulp'."
        ) from exc
    return pulp


def _solve_with_cbc(prob):
    """Solve the optimization problem with PuLP's CBC solver and verify the result."""
    pulp = _get_pulp()
    solver = None
    if hasattr(pulp, "PULP_CBC_CMD"):
        try:
            solver = pulp.PULP_CBC_CMD(msg=False)
        except Exception:
            solver = None

    if solver is None and hasattr(pulp, "COIN_CMD"):
        try:
            solver = pulp.COIN_CMD(msg=False)
        except Exception:
            solver = None

    try:
        status = prob.solve(solver) if solver else prob.solve()
    except Exception as exc:
        raise RuntimeError(
            "PuLP solver failed while solving the optimization model. "
            "Verify that a supported solver is installed and accessible to PuLP."
        ) from exc

    solver_status = pulp.LpStatus.get(status, "Unknown")
    if solver_status != "Optimal":
        raise RuntimeError(
            f"PuLP solver did not return an optimal result: {solver_status}. "
            "Check model feasibility and solver availability."
        )

    return status


def optimize_inventory(inventory_df, budget=None, holding_cost_rate=0.02, order_cost=150):
    """
    Classic EOQ (Economic Order Quantity) per SKU/region:
        EOQ = sqrt(2 * D * S / H)
    where D = annual demand, S = fixed order cost, H = annual holding cost per unit.
    If a total warehouse budget is supplied, orders are scaled down (MIP) to fit
    within budget, prioritizing SKUs with the highest stockout risk first.
    """
    df = inventory_df.copy()
    df["annual_demand"] = df["avg_daily_demand"] * 365
    df["holding_cost_per_unit"] = df["unit_cost"] * holding_cost_rate * 365
    df["eoq"] = np.sqrt(
        np.maximum(2 * df["annual_demand"] * order_cost / df["holding_cost_per_unit"].replace(0, np.nan), 0)
    ).fillna(0).round(0)

    df["recommended_order_qty"] = np.maximum(
        df["eoq"], df["reorder_point"] - df["units_on_hand"]
    ).clip(lower=0).round(0)
    df["order_cost_total"] = (df["recommended_order_qty"] * df["unit_cost"]).round(2)

    if budget is not None:
        pulp = _get_pulp()
        prob = pulp.LpProblem("inventory_budget_allocation", pulp.LpMaximize)
        # priority weight: stockout risk gets 3x weight so at-risk SKUs are funded first
        df["priority_weight"] = 1 + 2 * df["stockout_risk"]
        frac_vars = {i: pulp.LpVariable(f"frac_{i}", lowBound=0, upBound=1) for i in df.index}
        prob += pulp.lpSum(frac_vars[i] * df.loc[i, "recommended_order_qty"] * df.loc[i, "priority_weight"]
                            for i in df.index)
        prob += pulp.lpSum(frac_vars[i] * df.loc[i, "order_cost_total"] for i in df.index) <= budget
        _solve_with_cbc(prob)
        df["funded_fraction"] = [round(frac_vars[i].value() or 0, 3) for i in df.index]
        df["funded_order_qty"] = (df["recommended_order_qty"] * df["funded_fraction"]).round(0)
        df["funded_cost"] = (df["funded_order_qty"] * df["unit_cost"]).round(2)

    return df.sort_values("stockout_risk", ascending=False)


def optimize_price(base_price, base_demand, elasticity, cost_per_unit,
                    price_grid_pct=(-0.15, -0.10, -0.05, 0.0, 0.05, 0.08, 0.10, 0.15)):
    """
    Given a base price/demand point and a price elasticity of demand (%demand change
    per %price change, typically negative), evaluate a grid of candidate price changes
    and pick the one that maximizes total profit:
        demand(p) = base_demand * (1 + elasticity * pct_change)
        profit(p) = demand(p) * (price(p) - cost_per_unit)
    """
    rows = []
    for pct in price_grid_pct:
        new_price = base_price * (1 + pct)
        new_demand = max(base_demand * (1 + elasticity * pct), 0)
        revenue = new_price * new_demand
        profit = (new_price - cost_per_unit) * new_demand
        rows.append({
            "price_change_pct": pct, "price": round(new_price, 2),
            "estimated_demand": round(new_demand, 1), "estimated_revenue": round(revenue, 2),
            "estimated_profit": round(profit, 2),
        })
    grid = pd.DataFrame(rows)
    best = grid.loc[grid["estimated_profit"].idxmax()]
    return grid, best


def optimize_marketing_budget(region_predicted_roi, total_budget, min_per_region=0.0):
    """
    Linear program: allocate `total_budget` across regions to maximize predicted
    incremental revenue, where region_predicted_roi = {region: revenue per $1 spent}.
    Optionally enforce a minimum spend floor per region.
    """
    pulp = _get_pulp()
    regions = list(region_predicted_roi.keys())
    prob = pulp.LpProblem("marketing_allocation", pulp.LpMaximize)
    spend = {r: pulp.LpVariable(f"spend_{r}", lowBound=min_per_region) for r in regions}

    min_total = min_per_region * len(regions)
    if total_budget < min_total:
        raise ValueError(
            "Total marketing budget is too low to satisfy the per-region minimum spend floor. "
            f"Required at least ${min_total:,.2f}, got ${total_budget:,.2f}."
        )

    prob += pulp.lpSum(spend[r] * region_predicted_roi[r] for r in regions)
    prob += pulp.lpSum(spend[r] for r in regions) <= total_budget
    _solve_with_cbc(prob)

    allocation = {r: round(spend[r].value(), 2) for r in regions}
    predicted_revenue = {r: round(allocation[r] * region_predicted_roi[r], 2) for r in regions}
    return pd.DataFrame({
        "region": regions,
        "predicted_roi_per_dollar": [region_predicted_roi[r] for r in regions],
        "allocated_budget": [allocation[r] for r in regions],
        "predicted_incremental_revenue": [predicted_revenue[r] for r in regions],
    }).sort_values("allocated_budget", ascending=False)


def select_supplier(required_volume, suppliers, max_share=0.4):
    """
    suppliers: list of dicts {name, unit_cost, capacity}
    Chooses lowest-cost mix of suppliers to meet required_volume such that no single
    supplier exceeds `max_share` of total volume (diversification policy) -- Mixed
    Integer Program via PuLP.
    """
    pulp = _get_pulp()
    prob = pulp.LpProblem("supplier_selection", pulp.LpMinimize)
    qty = {s["name"]: pulp.LpVariable(f"qty_{s['name']}", lowBound=0, upBound=s["capacity"])
           for s in suppliers}

    max_total_supply = sum(min(s["capacity"], max_share * required_volume) for s in suppliers)
    if required_volume > max_total_supply:
        raise ValueError(
            "Required supplier volume cannot be met under the current max share diversification constraint. "
            f"Maximum feasible volume is {max_total_supply:.1f} but required volume is {required_volume:.1f}."
        )

    prob += pulp.lpSum(qty[s["name"]] * s["unit_cost"] for s in suppliers)
    prob += pulp.lpSum(qty[s["name"]] for s in suppliers) >= required_volume
    for s in suppliers:
        prob += qty[s["name"]] <= max_share * required_volume
    _solve_with_cbc(prob)

    rows = []
    for s in suppliers:
        q = qty[s["name"]].value() or 0
        rows.append({"supplier": s["name"], "unit_cost": s["unit_cost"], "capacity": s["capacity"],
                      "allocated_volume": round(q, 1), "allocated_cost": round(q * s["unit_cost"], 2),
                      "share_of_total": round(q / required_volume, 3) if required_volume else 0})
    return pd.DataFrame(rows).sort_values("allocated_volume", ascending=False)


if __name__ == "__main__":
    from pathlib import Path
    ROOT = Path(__file__).resolve().parent.parent
    inv = pd.read_parquet(ROOT / "feature_store" / "inventory_features.parquet")

    print("=== Inventory optimization (with $500,000 budget) ===")
    result = optimize_inventory(inv, budget=500_000)
    print(result[["region", "sku", "units_on_hand", "recommended_order_qty",
                  "funded_order_qty", "stockout_risk"]].head(10).to_string(index=False))

    print("\n=== Price optimization example (NovaFit Smartwatch, elasticity=-1.3) ===")
    grid, best = optimize_price(base_price=199.0, base_demand=1000, elasticity=-1.3, cost_per_unit=95.0)
    print(grid.to_string(index=False))
    print("\nBest:", best.to_dict())

    print("\n=== Marketing budget allocation ===")
    roi = {"North America": 3.2, "Europe": 1.8, "APAC": 4.1, "Latin America": 2.5}
    alloc = optimize_marketing_budget(roi, total_budget=1_000_000, min_per_region=50_000)
    print(alloc.to_string(index=False))

    print("\n=== Supplier selection (diversification-constrained) ===")
    suppliers = [
        {"name": "Nordic Supply Co", "unit_cost": 42.0, "capacity": 20000},
        {"name": "Meridian Components", "unit_cost": 45.0, "capacity": 25000},
        {"name": "Delta Manufacturing", "unit_cost": 47.5, "capacity": 15000},
    ]
    sel = select_supplier(required_volume=40000, suppliers=suppliers, max_share=0.4)
    print(sel.to_string(index=False))
