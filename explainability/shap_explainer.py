"""
UPI FraudShield — SHAP Explainability Module
Extracts top-3 SHAP feature contributions per transaction for alert cards.
Uses TreeExplainer (fast path) for real-time inference.
"""

import numpy as np
import pandas as pd
import shap
import joblib
import json
import os

EXPLAINER_PATH = os.path.join(os.path.dirname(__file__), "shap_explainer.pkl")
FEATURE_NAMES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "xgb_feature_names.json")

# Human-readable feature name mappings
FEATURE_DISPLAY_NAMES = {
    "amount": "Transaction Amount",
    "hour_of_day": "Time of Day",
    "is_new_device": "New Device",
    "is_weekend": "Weekend Transaction",
    "day_of_week": "Day of Week",
    "tx_count_2min": "Tx Count (2 min)",
    "tx_count_5min": "Tx Count (5 min)",
    "time_since_last_tx_capped": "Time Since Last Tx",
    "user_mean_amount": "User Avg Amount",
    "user_std_amount": "User Amount Variance",
    "amount_zscore": "Amount Z-Score",
    "amount_zscore_clipped": "Amount Z-Score (Clipped)",
    "hour_deviation": "Hour Deviation",
    "amount_ratio": "Amount vs User Avg",
    "amount_volatility": "Amount Volatility",
    "merchant_risk": "Merchant Risk Level",
    "app_risk": "UPI App Risk",
    "velocity_interaction": "Transaction Velocity",
    "is_late_night": "Late Night Transaction",
    "is_high_amount": "High Amount Flag",
    "new_device_velocity": "New Device + High Velocity",
    "new_device_high_amt": "New Device + High Amount",
    "merchant_category_enc": "Merchant Category",
    "upi_app_enc": "UPI App",
    "transaction_type_enc": "Transaction Type",
    "location_city_enc": "Location City",
}


def build_explainer(model, X_background: pd.DataFrame) -> shap.TreeExplainer:
    """
    Build SHAP TreeExplainer using a background sample.
    Uses 500 row background for speed; TreeExplainer is exact for tree models.
    """
    print("[SHAP] Building TreeExplainer...")
    # Use small background sample for speed (SHAP TreeExplainer doesn't need large background)
    background = X_background.sample(min(500, len(X_background)), random_state=42)
    explainer = shap.TreeExplainer(model, background)
    joblib.dump(explainer, EXPLAINER_PATH)
    print(f"[SHAP] Explainer saved → {EXPLAINER_PATH}")
    return explainer


def load_explainer() -> shap.TreeExplainer:
    """Load saved SHAP explainer."""
    if not os.path.exists(EXPLAINER_PATH):
        raise FileNotFoundError(f"Explainer not found at {EXPLAINER_PATH}. Run build_explainer() first.")
    return joblib.load(EXPLAINER_PATH)


def explain_transaction(X_row: pd.DataFrame, explainer: shap.TreeExplainer = None, top_k: int = 3) -> list:
    """
    Compute SHAP values for a single transaction row.
    Returns top-k feature contributions sorted by absolute SHAP value.

    Returns:
        List of dicts: [{"feature": str, "display_name": str, "shap_value": float, "direction": str}]
    """
    if explainer is None:
        explainer = load_explainer()

    shap_values = explainer.shap_values(X_row)

    # For binary classification, shap_values may be [neg_class, pos_class] or 2D
    if isinstance(shap_values, list):
        sv = shap_values[1][0]  # class 1 (fraud) SHAP values for first row
    else:
        sv = shap_values[0] if shap_values.ndim == 2 else shap_values

    feature_names = list(X_row.columns)

    # Sort by absolute SHAP value descending
    contributions = sorted(
        zip(feature_names, sv),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:top_k]

    results = []
    for feat_name, shap_val in contributions:
        results.append({
            "feature": feat_name,
            "display_name": FEATURE_DISPLAY_NAMES.get(feat_name, feat_name.replace("_", " ").title()),
            "shap_value": round(float(shap_val), 4),
            "direction": "increases_risk" if shap_val > 0 else "decreases_risk",
            "abs_value": round(abs(float(shap_val)), 4),
        })

    return results


def get_shap_summary(shap_features: list) -> dict:
    """
    Build a compact summary dict for the frontend alert card.
    """
    return {
        "top_features": [
            {
                "name": f["display_name"],
                "value": f["shap_value"],
                "direction": f["direction"],
                "bar_width_pct": round(min(f["abs_value"] * 200, 100), 1),  # scale for UI bar
            }
            for f in shap_features
        ]
    }


if __name__ == "__main__":
    # Test the explainer with a sample transaction
    from models.xgboost_classifier import load, build_features
    import pandas as pd

    print("Loading XGBoost model...")
    model, encoders = load()

    print("Loading test data...")
    df = pd.read_csv("upi_transactions.csv").head(1000)
    X, _, _ = build_features(df, fit_encoders=False, encoders=encoders)

    explainer = build_explainer(model, X)

    # Test on one transaction
    sample = X.iloc[[0]]
    result = explain_transaction(sample, explainer)
    print("\n=== SHAP TOP-3 FEATURES FOR SAMPLE TRANSACTION ===")
    for r in result:
        print(f"  {r['display_name']}: {r['shap_value']} ({r['direction']})")
