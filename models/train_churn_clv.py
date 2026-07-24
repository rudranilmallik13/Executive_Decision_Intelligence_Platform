"""
EDIP - Churn Prediction & Customer Lifetime Value (CLV) Models
================================================================
Churn model: XGBoost binary classifier predicting P(churn) per customer.
CLV model:   XGBoost regressor estimating projected 12-month revenue per customer,
             later discounted by churn risk in the decision engine.

Both models are explained with SHAP so any single customer's risk score can be
attributed to specific, human-readable factors.

Outputs (models/artifacts/):
    churn_model.joblib, churn_metrics.json
    clv_model.joblib, clv_metrics.json
    customer_scores.csv   (per-customer churn probability + CLV + top SHAP reasons)
"""
import json
import joblib
import numpy as np
import pandas as pd
import shap
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBClassifier, XGBRegressor

ROOT = Path(__file__).resolve().parent.parent
STORE = ROOT / "feature_store"
ARTIFACTS = ROOT / "models" / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

CHURN_FEATURES = [
    "tenure_days", "contract_length_months", "support_tickets_last_year",
    "satisfaction_score", "avg_order_value", "orders_last_year", "discount_pct",
    "late_payments_last_year", "is_high_risk_tickets", "is_short_contract",
]
CLV_FEATURES = CHURN_FEATURES + ["revenue_last_year_est"]
CAT_FEATURES = ["segment", "region"]


def prep(df):
    df = pd.get_dummies(df.copy(), columns=CAT_FEATURES, prefix=CAT_FEATURES)
    cat_cols = [c for c in df.columns if c.startswith(tuple(f"{c0}_" for c0 in CAT_FEATURES))]
    return df, cat_cols


def train_churn(df, feature_cols):
    X, y = df[feature_cols], df["churned"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.08,
        subsample=0.9, colsample_bytree=0.9, random_state=42,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    metrics = {
        "auc": float(roc_auc_score(y_test, proba)),
        "precision": float(precision_score(y_test, preds)),
        "recall": float(recall_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds)),
        "n_train": int(len(X_train)), "n_test": int(len(X_test)),
    }
    print(f"[churn] AUC={metrics['auc']:.3f} Precision={metrics['precision']:.2f} "
          f"Recall={metrics['recall']:.2f} F1={metrics['f1']:.2f}")

    explainer = shap.TreeExplainer(model)
    joblib.dump({"model": model, "feature_cols": feature_cols, "explainer": explainer},
                ARTIFACTS / "churn_model.joblib")
    with open(ARTIFACTS / "churn_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    return model, explainer


def train_clv(df, feature_cols):
    X = df[feature_cols]
    # proxy label: next-year revenue estimate scaled by retention likelihood (synthetic ground truth)
    y = df["revenue_last_year_est"] * (1.05 + np.random.normal(0, 0.08, len(df))) * (1 - 0.5 * df["churned"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.08,
                          subsample=0.9, colsample_bytree=0.9, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, preds)),
        "r2": float(r2_score(y_test, preds)),
        "n_train": int(len(X_train)), "n_test": int(len(X_test)),
    }
    print(f"[CLV] MAE=${metrics['mae']:.0f} R2={metrics['r2']:.3f}")

    joblib.dump({"model": model, "feature_cols": feature_cols}, ARTIFACTS / "clv_model.joblib")
    with open(ARTIFACTS / "clv_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    return model


def score_all_customers(df, churn_model, clv_model, churn_features, clv_features, explainer, raw_region, raw_segment):
    churn_proba = churn_model.predict_proba(df[churn_features])[:, 1]
    clv_pred = clv_model.predict(df[clv_features])

    shap_values = explainer.shap_values(df[churn_features])
    reasons = []
    for i in range(len(df)):
        row_shap = shap_values[i]
        top_idx = np.argsort(-np.abs(row_shap))[:3]
        reasons.append("; ".join(
            f"{churn_features[j]} ({'+' if row_shap[j] > 0 else '-'}{abs(row_shap[j]):.2f})"
            for j in top_idx
        ))

    out = pd.DataFrame({
        "customer_id": df["customer_id"].values,
        "region": raw_region.values,
        "segment": raw_segment.values,
        "churn_probability": churn_proba.round(3),
        "predicted_clv_12mo": clv_pred.round(2),
        "top_churn_drivers": reasons,
    }).sort_values("churn_probability", ascending=False)
    out.to_csv(ARTIFACTS / "customer_scores.csv", index=False)
    print(f"customer_scores.csv -> {len(out):,} rows written")
    return out


if __name__ == "__main__":
    raw = pd.read_parquet(STORE / "customer_features.parquet")
    df, cat_cols = prep(raw)
    churn_features = CHURN_FEATURES + cat_cols
    clv_features = CLV_FEATURES + cat_cols

    churn_model, explainer = train_churn(df, churn_features)
    clv_model = train_clv(df, clv_features)
    score_all_customers(df, churn_model, clv_model, churn_features, clv_features, explainer,
                         raw["region"], raw["segment"])

    print("\nModels saved to", ARTIFACTS)
