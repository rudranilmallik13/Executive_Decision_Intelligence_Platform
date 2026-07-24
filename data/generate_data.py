"""
EDIP - Synthetic Data Generator
================================
Generates a realistic, internally-consistent set of company datasets that feed the
Executive Decision Intelligence Platform:

  Structured (feeds the Feature Store / ML models):
    - erp_inventory.csv        SKU-level stock, lead time, reorder point, supplier
    - crm_customers.csv        customer master data, segment, contract, satisfaction
    - sales_transactions.csv   3 years of daily transactions (region x product x channel)
    - finance_statements.csv   monthly P&L line items by region
    - weather_daily.csv        daily weather by region (demand driver)
    - competitor_prices.csv    monthly competitor price index by product
    - market_news_sentiment.csv monthly macro / news sentiment score

  Unstructured (feeds the RAG / vector store):
    - docs/annual_report_2025.txt
    - docs/pricing_policy.txt
    - docs/q2_2026_board_meeting_notes.txt
    - docs/market_research_summary.txt
    - docs/supplier_policy.txt

Everything is generated with a fixed seed so the whole pipeline is reproducible.
A deliberate "Q2 2026 revenue drop" story is baked into the data (a price hike +
a competitor price war + a regional supply shortage) so the AI consultant has a
real root cause to discover later in the pipeline.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

np.random.seed(42)
DATA_DIR = Path(__file__).parent
DOCS_DIR = DATA_DIR / "docs"
DOCS_DIR.mkdir(exist_ok=True)

REGIONS = ["North America", "Europe", "APAC", "Latin America"]
CHANNELS = ["Online", "Retail Partner", "Direct B2B"]
PRODUCTS = [
    ("SKU-100", "Aurora Blender X1", "Kitchen Appliances", 89.0),
    ("SKU-101", "Aurora Blender X2 Pro", "Kitchen Appliances", 149.0),
    ("SKU-200", "NovaFit Smartwatch", "Wearables", 199.0),
    ("SKU-201", "NovaFit Smartwatch SE", "Wearables", 129.0),
    ("SKU-300", "CloudDesk Ergo Chair", "Office Furniture", 349.0),
    ("SKU-301", "CloudDesk Standing Desk", "Office Furniture", 429.0),
    ("SKU-400", "PulseAudio Earbuds", "Audio", 79.0),
    ("SKU-401", "PulseAudio Earbuds Max", "Audio", 159.0),
    ("SKU-500", "GreenHome Air Purifier", "Home Environment", 219.0),
    ("SKU-501", "GreenHome Air Purifier Mini", "Home Environment", 119.0),
]

START = datetime(2023, 1, 1)
END = datetime(2026, 6, 30)  # through Q2 2026 (the quarter the CEO is asking about)
all_days = pd.date_range(START, END, freq="D")

# ---------------------------------------------------------------------------
# 1. SALES TRANSACTIONS (daily, by region/product/channel)
# ---------------------------------------------------------------------------
rows = []
for d in all_days:
    month_idx = (d.year - START.year) * 12 + d.month
    weekday_factor = 1.15 if d.weekday() >= 5 else 1.0
    season_factor = 1.0 + 0.25 * np.sin(2 * np.pi * (d.timetuple().tm_yday / 365.0))
    # holiday bump in November/December
    holiday_factor = 1.6 if d.month in (11, 12) else 1.0

    for region in REGIONS:
        # --- baked-in Q2 2026 revenue-drop story ---
        region_shock = 1.0
        if region == "Europe" and d >= datetime(2026, 4, 1):
            region_shock = 0.62   # supply shortage: European DC lost a key supplier in Apr 2026
        if d >= datetime(2026, 4, 15):
            price_war_shock = 0.85  # competitor price war starting mid-April 2026 (all regions, mild)
        else:
            price_war_shock = 1.0

        for sku, name, category, base_price in PRODUCTS:
            price_hike_shock = 1.0
            price = base_price
            if d >= datetime(2026, 4, 1):
                # Company raised prices 8% on Kitchen Appliances & Wearables in Q2 2026
                if category in ("Kitchen Appliances", "Wearables"):
                    price = base_price * 1.08
                    price_hike_shock = 0.90  # demand elasticity response

            for channel in CHANNELS:
                channel_factor = {"Online": 1.0, "Retail Partner": 0.8, "Direct B2B": 0.5}[channel]
                lam = (
                    3.0 * channel_factor * weekday_factor * season_factor * holiday_factor
                    * region_shock * price_war_shock * price_hike_shock
                    * (1.0 + 0.03 * month_idx)  # gentle organic growth over time
                )
                units = np.random.poisson(max(lam, 0.05))
                if units == 0:
                    continue
                revenue = round(units * price, 2)
                cost = round(units * price * np.random.uniform(0.55, 0.68), 2)
                rows.append([d.date().isoformat(), region, channel, sku, name, category,
                             units, price, revenue, cost])

sales = pd.DataFrame(rows, columns=[
    "date", "region", "channel", "sku", "product_name", "category",
    "units_sold", "unit_price", "revenue", "cogs"
])
sales.to_csv(DATA_DIR / "sales_transactions.csv", index=False)
print(f"sales_transactions.csv -> {len(sales):,} rows")

# ---------------------------------------------------------------------------
# 2. FINANCE STATEMENTS (monthly P&L by region, derived from sales + opex assumptions)
# ---------------------------------------------------------------------------
sales["date"] = pd.to_datetime(sales["date"])
sales["month"] = sales["date"].values.astype("datetime64[M]")
fin = sales.groupby(["month", "region"]).agg(
    revenue=("revenue", "sum"),
    cogs=("cogs", "sum"),
).reset_index()
fin["gross_profit"] = fin["revenue"] - fin["cogs"]
fin["marketing_spend"] = (fin["revenue"] * np.random.uniform(0.06, 0.09, len(fin))).round(2)
fin["opex_other"] = (fin["revenue"] * np.random.uniform(0.10, 0.14, len(fin))).round(2)
fin["operating_income"] = fin["gross_profit"] - fin["marketing_spend"] - fin["opex_other"]
fin["month"] = fin["month"].dt.date.astype(str)
fin.to_csv(DATA_DIR / "finance_statements.csv", index=False)
print(f"finance_statements.csv -> {len(fin):,} rows")

# ---------------------------------------------------------------------------
# 3. CRM CUSTOMERS (with churn labels for the churn model)
# ---------------------------------------------------------------------------
N_CUST = 4000
cust_ids = [f"CUST-{i:05d}" for i in range(1, N_CUST + 1)]
signup_dates = pd.to_datetime(np.random.choice(pd.date_range("2021-01-01", "2026-01-01"), N_CUST))
segments = np.random.choice(["SMB", "Mid-Market", "Enterprise"], N_CUST, p=[0.55, 0.32, 0.13])
regions_c = np.random.choice(REGIONS, N_CUST)
contract_len = np.random.choice([1, 12, 24, 36], N_CUST, p=[0.35, 0.35, 0.20, 0.10])
tenure_days = (datetime(2026, 6, 30) - signup_dates).days
support_tickets = np.random.poisson(2.5, N_CUST)
satisfaction = np.clip(np.random.normal(7.5, 1.6, N_CUST), 1, 10).round(1)
avg_order_value = np.round(np.random.gamma(4, 45, N_CUST), 2)
orders_last_year = np.random.poisson(6, N_CUST)
discount_pct = np.round(np.random.uniform(0, 0.2, N_CUST), 3)
late_payments = np.random.poisson(0.4, N_CUST)

# churn probability driven by realistic factors -> generate a binary label
logit = (
    -1.6
    + 1.7 * (satisfaction < 6)
    + 1.5 * (support_tickets > 5)
    + 1.3 * (contract_len == 1)
    + 1.0 * (late_payments > 1)
    + 0.9 * (orders_last_year < 2)
    - 0.7 * (segments == "Enterprise")
    - 0.4 * (tenure_days / 365.0)
    + 0.15 * (10 - satisfaction)
    + np.random.normal(0, 0.35, N_CUST)
)
churn_prob = 1 / (1 + np.exp(-logit))
churned = (np.random.rand(N_CUST) < churn_prob).astype(int)

crm = pd.DataFrame({
    "customer_id": cust_ids,
    "region": regions_c,
    "segment": segments,
    "signup_date": signup_dates.date.astype(str),
    "contract_length_months": contract_len,
    "tenure_days": tenure_days,
    "support_tickets_last_year": support_tickets,
    "satisfaction_score": satisfaction,
    "avg_order_value": avg_order_value,
    "orders_last_year": orders_last_year,
    "discount_pct": discount_pct,
    "late_payments_last_year": late_payments,
    "churned": churned,
})
crm.to_csv(DATA_DIR / "crm_customers.csv", index=False)
print(f"crm_customers.csv -> {len(crm):,} rows, churn rate={crm['churned'].mean():.2%}")

# ---------------------------------------------------------------------------
# 4. ERP INVENTORY
# ---------------------------------------------------------------------------
erp_rows = []
suppliers = ["Meridian Components", "PacificParts Ltd", "Nordic Supply Co", "Delta Manufacturing"]
for region in REGIONS:
    for sku, name, category, price in PRODUCTS:
        on_hand = int(np.random.uniform(200, 3000))
        lead_time = int(np.random.choice([7, 14, 21, 30]))
        daily_demand_est = sales[(sales.region == region) & (sales.sku == sku)]["units_sold"].mean()
        daily_demand_est = 0 if np.isnan(daily_demand_est) else daily_demand_est
        safety_stock = round(daily_demand_est * lead_time * 0.5, 1)
        reorder_point = round(daily_demand_est * lead_time + safety_stock, 1)
        erp_rows.append([region, sku, name, category, on_hand, lead_time,
                          round(daily_demand_est, 2), safety_stock, reorder_point,
                          np.random.choice(suppliers), round(price * np.random.uniform(0.5, 0.65), 2)])
erp = pd.DataFrame(erp_rows, columns=[
    "region", "sku", "product_name", "category", "units_on_hand", "supplier_lead_time_days",
    "avg_daily_demand", "safety_stock", "reorder_point", "supplier", "unit_cost"
])
erp.to_csv(DATA_DIR / "erp_inventory.csv", index=False)
print(f"erp_inventory.csv -> {len(erp):,} rows")

# ---------------------------------------------------------------------------
# 5. WEATHER (daily, by region) -- external demand driver
# ---------------------------------------------------------------------------
weather_rows = []
for d in all_days:
    for region in REGIONS:
        base_temp = {"North America": 15, "Europe": 12, "APAC": 26, "Latin America": 24}[region]
        temp = base_temp + 10 * np.sin(2 * np.pi * d.timetuple().tm_yday / 365.0) + np.random.normal(0, 3)
        precip = max(0, np.random.gamma(1.5, 3) - 2)
        weather_rows.append([d.date().isoformat(), region, round(temp, 1), round(precip, 1)])
weather = pd.DataFrame(weather_rows, columns=["date", "region", "avg_temp_c", "precipitation_mm"])
weather.to_csv(DATA_DIR / "weather_daily.csv", index=False)
print(f"weather_daily.csv -> {len(weather):,} rows")

# ---------------------------------------------------------------------------
# 6. COMPETITOR PRICES (monthly index per category)
# ---------------------------------------------------------------------------
months = pd.date_range(START, END, freq="MS")
comp_rows = []
categories = sorted(set(c for _, _, c, _ in PRODUCTS))
for m in months:
    for cat in categories:
        idx = 100 + np.random.normal(0, 2)
        if m >= datetime(2026, 4, 1):
            idx -= 12  # competitor price war starting Apr 2026
        comp_rows.append([m.date().isoformat(), cat, round(idx, 1)])
competitor = pd.DataFrame(comp_rows, columns=["month", "category", "competitor_price_index"])
competitor.to_csv(DATA_DIR / "competitor_prices.csv", index=False)
print(f"competitor_prices.csv -> {len(competitor):,} rows")

# ---------------------------------------------------------------------------
# 7. MARKET NEWS SENTIMENT (monthly macro sentiment, -1..1)
# ---------------------------------------------------------------------------
news_rows = []
sentiment = 0.15
for m in months:
    sentiment += np.random.normal(0, 0.08)
    if m >= datetime(2026, 4, 1):
        sentiment -= 0.05  # inflation worries drag sentiment down in Q2 2026
    sentiment = float(np.clip(sentiment, -1, 1))
    inflation_rate = 2.8 + (1.1 if m >= datetime(2026, 4, 1) else 0) + np.random.normal(0, 0.15)
    news_rows.append([m.date().isoformat(), round(sentiment, 3), round(inflation_rate, 2)])
news = pd.DataFrame(news_rows, columns=["month", "market_sentiment_score", "inflation_rate_pct"])
news.to_csv(DATA_DIR / "market_news_sentiment.csv", index=False)
print(f"market_news_sentiment.csv -> {len(news):,} rows")

# ---------------------------------------------------------------------------
# 8. UNSTRUCTURED DOCUMENTS (for the RAG pipeline)
# ---------------------------------------------------------------------------
(DOCS_DIR / "annual_report_2025.txt").write_text("""ANNUAL REPORT - FISCAL YEAR 2025
Company: Aurora Home & Wearables Inc.

EXECUTIVE SUMMARY
Fiscal year 2025 was a year of steady growth for the company. Total revenue grew 11%
year over year, driven primarily by strong performance in the Wearables and Kitchen
Appliances categories. The North America and APAC regions were the fastest-growing
regions, each posting double-digit revenue growth. Europe grew more modestly at 4%,
constrained by softer consumer demand and a highly competitive pricing environment.

PRODUCT PERFORMANCE
The NovaFit Smartwatch line remained the single largest revenue contributor, benefiting
from a successful mid-year firmware update that improved battery life and expanded
into three new retail partnerships. The Aurora Blender line saw renewed demand after
a redesign of the X2 Pro model. CloudDesk office furniture continued to normalize
after the post-pandemic remote-work furniture boom faded.

SUPPLY CHAIN
The company relies on four primary contract manufacturers: Meridian Components,
PacificParts Ltd, Nordic Supply Co, and Delta Manufacturing. Nordic Supply Co, which
handles roughly 30% of European fulfillment volume for Kitchen Appliances and
Wearables, has flagged capacity constraints for calendar year 2026 due to a facility
consolidation. Management is evaluating a secondary supplier in the region to
mitigate concentration risk.

OUTLOOK FOR 2026
Management expects continued growth but flagged three risks for the year ahead:
(1) input cost inflation, particularly for electronics components, (2) intensifying
price competition in the wearables and kitchen appliance categories as new entrants
enter the market, and (3) potential supply disruption in Europe tied to the Nordic
Supply Co capacity issue noted above. The Board approved a pricing review for Q2 2026
to protect gross margin against rising input costs, to be implemented selectively in
the Kitchen Appliances and Wearables categories.
""")

(DOCS_DIR / "pricing_policy.txt").write_text("""INTERNAL PRICING POLICY
Document owner: VP of Pricing & Revenue Strategy
Last updated: March 2026

1. PRICING REVIEW CADENCE
Prices are reviewed quarterly. Any price change above 5% requires approval from the
Pricing Committee and must include an elasticity estimate and a projected margin
impact model.

2. Q2 2026 PRICE ADJUSTMENT
Effective April 1, 2026, list prices for the Kitchen Appliances and Wearables
categories were increased by 8% globally to offset rising input costs (electronics
components, freight). This was approved by the Pricing Committee on March 18, 2026.
The Committee's internal elasticity model projected a 9-11% unit volume decline in
the affected categories following the increase, partially offset by higher revenue
per unit. The model did NOT account for a simultaneous competitor price war, which
began in mid-April 2026 and was not anticipated at the time of approval.

3. DISCOUNTING GUIDELINES
Sales representatives may offer discounts of up to 10% without additional approval.
Discounts between 10-20% require regional sales director approval. Discounts above
20% require VP approval and must be logged with a business justification.

4. COMPETITIVE RESPONSE PROTOCOL
If competitor pricing drops more than 8% in a category within a 30-day window, the
Pricing Committee will convene an emergency review within 5 business days to
evaluate a response (price match, promotional bundle, or hold).
""")

(DOCS_DIR / "q2_2026_board_meeting_notes.txt").write_text("""BOARD MEETING NOTES - Q2 2026 BUSINESS REVIEW
Date: July 8, 2026 (Confidential - Internal Use Only)

ATTENDEES: CEO, CFO, VP Sales, VP Supply Chain, VP Pricing, Board Members

1. REVENUE PERFORMANCE
Q2 2026 revenue came in below plan. The CFO presented a regional breakdown showing
that Europe was the primary driver of the shortfall, with revenue down sharply
compared to Q1. Two compounding factors were identified:
  (a) A supplier disruption at Nordic Supply Co's European facility reduced
      available inventory for Kitchen Appliances and Wearables SKUs starting in
      April, well below the original 2026 outlook flagged in the Annual Report.
  (b) The April price increase, while approved and modeled internally, coincided
      with an unanticipated competitor price war in the same categories, amplifying
      the expected unit volume decline beyond the Pricing Committee's original
      elasticity estimate.

2. CUSTOMER IMPACT
The VP of Sales noted a rise in customer complaints related to order delays in
Europe, and early churn-model signals showing elevated risk among Mid-Market
customers with high support-ticket volume in that region.

3. ACTION ITEMS
  - VP Supply Chain to finalize a secondary European supplier agreement by end of Q3 2026.
  - VP Pricing to run a competitive-response scenario analysis on a partial price
    rollback for the affected categories in Europe only.
  - VP Sales to prioritize retention outreach to at-risk Mid-Market accounts in Europe.
  - CFO to update the FY2026 revenue forecast reflecting the Q2 shortfall and the
    supplier resolution timeline.

4. INFLATION SENSITIVITY
The Board requested a scenario analysis on the impact of a further 3-percentage-point
increase in input cost inflation on gross margin and recommended pricing actions,
to be presented at the Q3 2026 board meeting.
""")

(DOCS_DIR / "market_research_summary.txt").write_text("""MARKET RESEARCH SUMMARY - Consumer Electronics & Home Categories
Prepared by: External Research Partner, June 2026

KEY FINDINGS
- Two new low-cost entrants launched competing wearable devices in Q1 2026,
  triggering an industry-wide price war that intensified in April 2026.
- Consumer sentiment toward premium kitchen appliances has softened slightly amid
  broader inflation concerns, though demand remains resilient in the Enterprise
  and gifting segments.
- Smart home and air purification categories are forecast to grow at a
  high-single-digit CAGR through 2028, outpacing legacy kitchen appliance growth.
- Consumers report increasing price sensitivity in Europe specifically, more so
  than in North America or APAC, which may amplify the impact of list price
  increases in that region.
""")

(DOCS_DIR / "supplier_policy.txt").write_text("""SUPPLIER MANAGEMENT POLICY
Document owner: VP Supply Chain

1. SUPPLIER DIVERSIFICATION
No single supplier should account for more than 40% of regional fulfillment volume
for any product category. Nordic Supply Co currently accounts for approximately 30%
of European fulfillment volume for Kitchen Appliances and Wearables, within policy
limits but flagged as a concentration risk given its 2026 capacity constraints.

2. LEAD TIME AND SAFETY STOCK
Standard safety stock is set at 50% of expected demand over the supplier lead time.
Regions with a single dominant supplier for a category should hold safety stock at
the higher end of this range (60-70%) to buffer against disruption.

3. SUPPLIER EVALUATION CRITERIA
Suppliers are evaluated quarterly on: on-time delivery rate, defect rate, cost
competitiveness, and capacity headroom. A secondary supplier must be qualified for
any product category exceeding $5M in annual regional volume from a single source.
""")

print("Unstructured documents written to data/docs/")
print("\nAll datasets generated successfully.")
