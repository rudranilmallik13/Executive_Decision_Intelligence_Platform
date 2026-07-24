"""
EDIP - Data Pipeline / Feature Store Builder
=============================================
Ingests all raw sources (ERP, CRM, Sales, Finance, Weather, Competitor Prices,
Market News) and produces clean, model-ready feature tables:

    feature_store/monthly_region_features.parquet   -> for demand & revenue forecasting
    feature_store/customer_features.parquet         -> for churn & CLV models
    feature_store/inventory_features.parquet         -> for inventory optimization

Run:  python3 pipeline/build_feature_store.py
"""
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
STORE = ROOT / "feature_store"
STORE.mkdir(exist_ok=True)


def load_raw():
    sales = pd.read_csv(DATA / "sales_transactions.csv", parse_dates=["date"])
    finance = pd.read_csv(DATA / "finance_statements.csv", parse_dates=["month"])
    crm = pd.read_csv(DATA / "crm_customers.csv", parse_dates=["signup_date"])
    erp = pd.read_csv(DATA / "erp_inventory.csv")
    weather = pd.read_csv(DATA / "weather_daily.csv", parse_dates=["date"])
    competitor = pd.read_csv(DATA / "competitor_prices.csv", parse_dates=["month"])
    news = pd.read_csv(DATA / "market_news_sentiment.csv", parse_dates=["month"])
    return sales, finance, crm, erp, weather, competitor, news


def build_monthly_region_features(sales, finance, weather, competitor, news):
    sales = sales.copy()
    sales["month"] = sales["date"].values.astype("datetime64[M]")

    monthly_sales = sales.groupby(["month", "region"]).agg(
        revenue=("revenue", "sum"),
        units_sold=("units_sold", "sum"),
        cogs=("cogs", "sum"),
        avg_unit_price=("unit_price", "mean"),
    ).reset_index()

    monthly_sales["month_dt"] = monthly_sales["month"]

    weather_m = weather.copy()
    weather_m["month"] = weather_m["date"].values.astype("datetime64[M]")
    weather_m = weather_m.groupby(["month", "region"]).agg(
        avg_temp_c=("avg_temp_c", "mean"),
        total_precip_mm=("precipitation_mm", "sum"),
    ).reset_index()

    comp_m = competitor.groupby("month").agg(
        competitor_price_index=("competitor_price_index", "mean")
    ).reset_index()

    df = monthly_sales.merge(weather_m, on=["month", "region"], how="left")
    df = df.merge(comp_m, on="month", how="left")
    df = df.merge(news, on="month", how="left")
    df = df.merge(finance[["month", "region", "marketing_spend", "opex_other", "operating_income"]],
                   on=["month", "region"], how="left")

    df = df.sort_values(["region", "month"]).reset_index(drop=True)
    # lag features (previous month revenue/units) per region -> classic forecasting features
    for col in ["revenue", "units_sold"]:
        df[f"{col}_lag1"] = df.groupby("region")[col].shift(1)
        df[f"{col}_lag2"] = df.groupby("region")[col].shift(2)
        df[f"{col}_rolling3"] = df.groupby("region")[col].transform(lambda s: s.shift(1).rolling(3).mean())

    df["month_num"] = df["month"].dt.month
    df["year"] = df["month"].dt.year
    df["time_idx"] = (df["year"] - df["year"].min()) * 12 + df["month_num"]

    df = df.dropna(subset=["revenue_lag1"]).reset_index(drop=True)
    df.to_parquet(STORE / "monthly_region_features.parquet", index=False)
    df.to_csv(STORE / "monthly_region_features.csv", index=False)
    print(f"monthly_region_features -> {df.shape}")
    return df


def build_customer_features(crm):
    df = crm.copy()
    df["orders_per_month"] = df["orders_last_year"] / 12.0
    df["is_high_risk_tickets"] = (df["support_tickets_last_year"] > 5).astype(int)
    df["is_short_contract"] = (df["contract_length_months"] <= 1).astype(int)
    df["revenue_last_year_est"] = df["avg_order_value"] * df["orders_last_year"]
    df.to_parquet(STORE / "customer_features.parquet", index=False)
    df.to_csv(STORE / "customer_features.csv", index=False)
    print(f"customer_features -> {df.shape}")
    return df


def build_inventory_features(erp):
    df = erp.copy()
    df["days_of_supply"] = np.where(df["avg_daily_demand"] > 0,
                                     df["units_on_hand"] / df["avg_daily_demand"], np.inf)
    df["stockout_risk"] = (df["days_of_supply"] < df["supplier_lead_time_days"]).astype(int)
    df.to_parquet(STORE / "inventory_features.parquet", index=False)
    df.to_csv(STORE / "inventory_features.csv", index=False)
    print(f"inventory_features -> {df.shape}")
    return df


if __name__ == "__main__":
    sales, finance, crm, erp, weather, competitor, news = load_raw()
    build_monthly_region_features(sales, finance, weather, competitor, news)
    build_customer_features(crm)
    build_inventory_features(erp)
    print("\nFeature store built at:", STORE)
