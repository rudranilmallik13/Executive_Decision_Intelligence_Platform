"""
EDIP - Explainable AI Layer
==============================
Every recommendation the platform makes should come with:
    - a confidence score
    - the specific data that supports it
    - the top model features (SHAP values) driving the prediction
    - the source documents (if RAG was used)
    - the assumptions made

This module wraps SHAP explanations into a single, structured "explanation card"
that the agent (agent/ai_consultant.py) attaches to every answer.
"""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "models" / "artifacts"


def _get_shap():
    try:
        import shap
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "SHAP is required for explainability functions. Install it with 'python -m pip install shap'."
        ) from exc
    return shap


def explain_revenue_forecast(region_row: pd.DataFrame, top_n=5):
    """Returns a ranked list of {feature, value, shap_contribution} for a single
    region-month's revenue forecast, plus a naive confidence score derived from
    the model's out-of-sample R^2 (stored at training time)."""
    try:
        shap = _get_shap()
    except ModuleNotFoundError:
        bundle = joblib.load(ARTIFACTS / "revenue_forecast_model.joblib")
        model, feature_cols = bundle["model"], bundle["feature_cols"]
        prediction = float(model.predict(region_row[feature_cols])[0])

        import json
        metrics_path = ARTIFACTS / "revenue_forecast_model_metrics.json"
        r2 = None
        if metrics_path.exists():
            with open(metrics_path) as f:
                r2 = json.load(f).get("r2")

        return {
            "prediction": round(prediction, 2),
            "base_value": None,
            "confidence_score": round(float(r2), 3) if r2 is not None else None,
            "top_contributing_factors": [],
            "explanation_unavailable": "SHAP is not installed; install shap to enable model explanations.",
        }

    bundle = joblib.load(ARTIFACTS / "revenue_forecast_model.joblib")
    model, feature_cols = bundle["model"], bundle["feature_cols"]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(region_row[feature_cols])[0]
    base_value = explainer.expected_value

    contributions = sorted(
        zip(feature_cols, region_row[feature_cols].iloc[0].values, shap_values),
        key=lambda t: -abs(t[2])
    )[:top_n]

    prediction = float(model.predict(region_row[feature_cols])[0])

    import json
    metrics_path = ARTIFACTS / "revenue_forecast_model_metrics.json"
    r2 = None
    if metrics_path.exists():
        with open(metrics_path) as f:
            r2 = json.load(f).get("r2")

    return {
        "prediction": round(prediction, 2),
        "base_value": round(float(base_value), 2),
        "confidence_score": round(float(r2), 3) if r2 is not None else None,
        "top_contributing_factors": [
            {"feature": f, "value": round(float(v), 2), "shap_contribution": round(float(s), 2),
             "direction": "increases" if s > 0 else "decreases"}
            for f, v, s in contributions
        ],
    }


def explain_churn_prediction(customer_row_features: pd.DataFrame, top_n=5):
    bundle = joblib.load(ARTIFACTS / "churn_model.joblib")
    model, feature_cols = bundle["model"], bundle["feature_cols"]
    try:
        shap = _get_shap()
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(customer_row_features[feature_cols])
        # binary classifier: shap_values shape (n, features) for the positive class in newer shap versions
        sv = shap_values[0] if shap_values.ndim == 1 else shap_values[0]
    except ModuleNotFoundError:
        proba = float(model.predict_proba(customer_row_features[feature_cols])[0, 1])
        return {
            "churn_probability": round(proba, 3),
            "risk_tier": "High" if proba > 0.6 else ("Medium" if proba > 0.3 else "Low"),
            "top_contributing_factors": [],
            "explanation_unavailable": "SHAP is not installed; install shap to enable churn explanations.",
        }

    proba = float(model.predict_proba(customer_row_features[feature_cols])[0, 1])
    contributions = sorted(
        zip(feature_cols, customer_row_features[feature_cols].iloc[0].values, sv),
        key=lambda t: -abs(t[2])
    )[:top_n]

    return {
        "churn_probability": round(proba, 3),
        "risk_tier": "High" if proba > 0.6 else ("Medium" if proba > 0.3 else "Low"),
        "top_contributing_factors": [
            {"feature": f, "value": round(float(v), 2), "shap_contribution": round(float(s), 2),
             "direction": "increases risk" if s > 0 else "decreases risk"}
            for f, v, s in contributions
        ],
    }


def confidence_from_metrics(metrics: dict) -> str:
    """Turns raw model metrics into a plain-language confidence label for executives."""
    r2 = metrics.get("r2")
    auc = metrics.get("auc")
    score = r2 if r2 is not None else auc
    if score is None:
        return "Unknown"
    if score >= 0.85:
        return "High"
    if score >= 0.7:
        return "Medium"
    return "Low"


if __name__ == "__main__":
    import pandas as pd
    STORE = ROOT / "feature_store"
    df = pd.read_parquet(STORE / "monthly_region_features.parquet")
    df_dummy = pd.get_dummies(df, columns=["region"], prefix="region")
    europe_latest = df_dummy[df_dummy.get("region_Europe", 0) == 1].sort_values("time_idx").iloc[[-1]]

    explanation = explain_revenue_forecast(europe_latest)
    print("Revenue forecast explanation (Europe, latest month):")
    for k, v in explanation.items():
        print(f"  {k}: {v}")
