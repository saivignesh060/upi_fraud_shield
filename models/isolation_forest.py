"""
UPI FraudShield — Isolation Forest Anomaly Detector (Layer 2)
Unsupervised anomaly detection without requiring labels.
Detects novel/unknown fraud patterns based on statistical deviation.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
import joblib
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "isolation_forest.pkl")
ENCODERS_PATH = os.path.join(os.path.dirname(__file__), "if_encoders.pkl")

# ─────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────
FEATURES = [
    "amount_zscore",       # Z-score of tx amount vs user baseline
    "hour_deviation",      # Deviation from user's typical active hours
    "tx_count_2min",       # Tx velocity: count in last 2 minutes
    "tx_count_5min",       # Tx velocity: count in last 5 minutes
    "merchant_cat_enc",    # Encoded merchant category (proxy for embedding)
    "amount",              # Raw amount
    "is_new_device",       # New device flag
    "is_weekend",          # Weekend indicator
    "day_of_week",         # Day of week
    "hour_of_day",         # Hour of day
    "time_since_last_tx",  # Seconds since last transaction (capped)
]

# Merchant risk scores — higher = riskier category
MERCHANT_RISK = {
    "groceries": 0.1,      "food_delivery": 0.1,   "utilities": 0.1,
    "fuel": 0.15,          "clothing": 0.15,         "healthcare": 0.1,
    "education": 0.1,      "insurance": 0.1,          "mutual_funds": 0.2,
    "recharge": 0.15,      "electronics": 0.2,        "travel": 0.25,
    "entertainment": 0.2,  "crypto_exchange": 0.9,   "gambling": 0.95,
    "p2p_transfer": 0.3,
}


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features for Isolation Forest from raw transaction data.
    """
    feat = df.copy()

    # Convert timestamp if needed
    if feat["timestamp"].dtype == object:
        feat["timestamp"] = pd.to_datetime(feat["timestamp"])

    # Hour deviation: |current_hour - user's mean active hour|
    # Use midpoint of active window (derived from hour_of_day baseline in data)
    user_typical_hour = feat.groupby("user_id")["hour_of_day"].transform("mean")
    feat["hour_deviation"] = (feat["hour_of_day"] - user_typical_hour).abs()

    # Merchant category encoding (risk-weighted numeric)
    feat["merchant_cat_enc"] = feat["merchant_category"].map(MERCHANT_RISK).fillna(0.3)

    # Cap time_since_last_tx at 86400 (1 day) to prevent outlier distortion
    feat["time_since_last_tx"] = feat["time_since_last_tx"].clip(upper=86400)

    # Ensure is_new_device is numeric
    feat["is_new_device"] = feat["is_new_device"].astype(int)

    return feat[FEATURES]


def train(df: pd.DataFrame, contamination: float = 0.032) -> IsolationForest:
    """
    Train Isolation Forest on full dataset.
    Contamination = PRD fraud rate (3.2%).
    """
    print("[IF] Building feature matrix...")
    X = build_features(df)

    print(f"[IF] Training Isolation Forest (contamination={contamination}, n_estimators=200)...")
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        max_samples="auto",
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )
    model.fit(X)

    # Save model
    joblib.dump(model, MODEL_PATH)
    print(f"[IF] Model saved → {MODEL_PATH}")

    # Quick evaluation
    scores = model.decision_function(X)
    predictions = model.predict(X)  # -1=anomaly, 1=normal
    anomaly_count = (predictions == -1).sum()
    print(f"[IF] Anomalies flagged: {anomaly_count:,} ({anomaly_count/len(X)*100:.2f}%)")

    return model


def load() -> IsolationForest:
    """Load saved Isolation Forest model."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train() first.")
    return joblib.load(MODEL_PATH)


def score_transaction(txn: dict, model: IsolationForest = None) -> float:
    """
    Score a single transaction. Returns risk contribution (0–25 pts).
    anomaly_score from IF: negative = more anomalous.
    We invert and normalize to 0-25 scale.
    """
    if model is None:
        model = load()

    row = pd.DataFrame([txn])
    # Fill missing engineered features with safe defaults
    defaults = {
        "amount_zscore": 0.0,
        "hour_deviation": 0.0,
        "tx_count_2min": 0,
        "tx_count_5min": 0,
        "merchant_cat_enc": 0.3,
        "amount": txn.get("amount", 0),
        "is_new_device": int(txn.get("is_new_device", False)),
        "is_weekend": 0,
        "day_of_week": 0,
        "hour_of_day": txn.get("hour_of_day", 12),
        "time_since_last_tx": min(txn.get("time_since_last_tx", 99999), 86400),
    }
    for k, v in defaults.items():
        if k not in row.columns:
            row[k] = v

    # Merchant category risk
    if "merchant_category" in txn:
        row["merchant_cat_enc"] = MERCHANT_RISK.get(txn["merchant_category"], 0.3)

    X = row[FEATURES]
    raw_score = model.decision_function(X)[0]  # More negative = more anomalous

    # Normalize: typical range is [-0.2, 0.1]. Map to 0-25 pts
    # raw_score <= -0.2 → 25 pts; raw_score >= 0.1 → 0 pts
    normalized = np.clip((-raw_score - (-0.2)) / (0.1 - (-0.2)), 0, 1)
    risk_pts = round(float(normalized * 25), 2)
    return risk_pts


def evaluate(df: pd.DataFrame, model: IsolationForest = None) -> dict:
    """
    Evaluate IF model on labeled dataset.
    Since IF is unsupervised, we check its fraud detection rate.
    """
    if model is None:
        model = load()

    X = build_features(df)
    predictions = model.predict(X)  # -1=anomaly, 1=normal
    y_true = df["fraud_label"].values

    # Map: IF -1 → 1 (fraud predicted), IF 1 → 0 (legit)
    y_pred = (predictions == -1).astype(int)

    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    metrics = {
        "model": "Isolation Forest",
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "false_positive_rate": round(fpr, 4),
    }
    return metrics


if __name__ == "__main__":
    print("Loading dataset...")
    df = pd.read_csv("upi_transactions.csv")
    model = train(df)
    metrics = evaluate(df, model)
    print("\n=== ISOLATION FOREST METRICS ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
