"""
EDIP - Revenue & Demand Forecast Models
=========================================
Trains gradient-boosted regression models (XGBoost) to forecast next-month
revenue and unit demand per region, using lag features, seasonality, weather,
competitor pricing and macro sentiment as predictors.

Also computes SHAP values so every forecast can be explained ("why did the
model predict this?").

Outputs (models/artifacts/):
    revenue_forecast_model.joblib
    demand_forecast_model.joblib
    revenue_forecast_metrics.json
    demand_forecast_metrics.json
"""
import json
import joblib
import numpy as np
import pandas as pd
import shap
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent.parent
STORE = ROOT / "feature_store"
ARTIFACTS = ROOT / "models" / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "revenue_lag1", "revenue_lag2", "revenue_rolling3",
    "units_sold_lag1", "units_sold_lag2", "units_sold_rolling3",
    "avg_temp_c", "total_precip_mm", "competitor_price_index",
    "market_sentiment_score", "inflation_rate_pct",
    "marketing_spend", "month_num", "time_idx",
]
CAT_FEATURES = ["region"]


def prep(df):
    df = df.copy()
    df = pd.get_dummies(df, columns=CAT_FEATURES, prefix="region")
    region_cols = [c for c in df.columns if c.startswith("region_")]
    feature_cols = FEATURES + region_cols
    return df, feature_cols


def train_target(df, feature_cols, target, out_name):
    X = df[feature_cols]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBRegressor(
        n_estimators=250, max_depth=4, learning_rate=0.06,
        subsample=0.9, colsample_bytree=0.9, random_state=42,
        reg_lambda=1.0,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    metrics = {
        "target": target,
        "mae": float(mean_absolute_error(y_test, preds)),
        "mape": float(mean_absolute_percentage_error(y_test, preds)),
        "r2": float(r2_score(y_test, preds)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    print(f"[{target}] MAE={metrics['mae']:.1f} MAPE={metrics['mape']:.2%} R2={metrics['r2']:.3f}")

    # SHAP explainability - global feature importance
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance = sorted(zip(feature_cols, mean_abs_shap.tolist()), key=lambda t: -t[1])
    metrics["top_drivers"] = [{"feature": f, "mean_abs_shap": round(v, 2)} for f, v in importance[:8]]

    joblib.dump({"model": model, "feature_cols": feature_cols}, ARTIFACTS / f"{out_name}.joblib")
    with open(ARTIFACTS / f"{out_name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    return model, metrics


if __name__ == "__main__":
    df = pd.read_parquet(STORE / "monthly_region_features.parquet")
    df_prepped, feature_cols = prep(df)

    train_target(df_prepped, feature_cols, "revenue", "revenue_forecast_model")
    train_target(df_prepped, feature_cols, "units_sold", "demand_forecast_model")

    print("\nModels saved to", ARTIFACTS)
