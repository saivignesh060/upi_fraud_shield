"""
UPI FraudShield — XGBoost Supervised Fraud Classifier (Layer 3)
Trained on labeled synthetic data to classify transactions with high precision.
Target: Precision ≥ 88%, Recall ≥ 85%, F1 ≥ 86%, ROC-AUC ≥ 0.93, FPR ≤ 2%
"""

import numpy as np
import pandas as pd
import joblib
import os
import json
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.preprocessing import LabelEncoder


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


MODEL_PATH = os.path.join(os.path.dirname(__file__), "xgboost_classifier.pkl")
ENCODERS_PATH = os.path.join(os.path.dirname(__file__), "xgb_encoders.pkl")
METRICS_PATH = os.path.join(os.path.dirname(__file__), "xgb_metrics.json")
FEATURE_NAMES_PATH = os.path.join(os.path.dirname(__file__), "xgb_feature_names.json")

# ─────────────────────────────────────────────────────────────
# SCORING THRESHOLDS
# DISPLAY_THRESHOLD : fraud_prob >= 0.40 → flag in live feed (yellow alert)
# BLOCK_THRESHOLD   : fraud_prob >= 0.90 → hard block + red alert card
# ─────────────────────────────────────────────────────────────
DISPLAY_THRESHOLD = 0.40   # Transaction appears as flagged in the live feed
BLOCK_THRESHOLD   = 0.90   # Transaction is actually blocked

# ─────────────────────────────────────────────────────────────
# MERCHANT RISK SCORES (consistent with isolation_forest.py)
# ─────────────────────────────────────────────────────────────
MERCHANT_RISK = {
    "groceries": 0.1,       "food_delivery": 0.1,   "utilities": 0.1,
    "fuel": 0.15,           "clothing": 0.15,         "healthcare": 0.1,
    "education": 0.1,       "insurance": 0.1,          "mutual_funds": 0.2,
    "recharge": 0.15,       "electronics": 0.2,        "travel": 0.25,
    "entertainment": 0.2,   "crypto_exchange": 0.9,   "gambling": 0.95,
    "p2p_transfer": 0.3,
}

UPI_APP_RISK = {
    "GPay": 0.1, "PhonePe": 0.1, "Paytm": 0.15, "BHIM": 0.1,
    "AmazonPay": 0.1, "WhatsApp Pay": 0.12,
}


def build_features(df: pd.DataFrame, fit_encoders: bool = False, encoders: dict = None) -> tuple:
    """
    Build the full feature matrix for XGBoost.
    Includes all schema fields + 18 engineered features.
    Returns: (X DataFrame, encoders dict, feature_col_names)
    """
    feat = df.copy()

    # Parse timestamp
    if feat["timestamp"].dtype == object:
        feat["timestamp"] = pd.to_datetime(feat["timestamp"])

    # ── ENGINEERED FEATURES ────────────────────────────────
    # 1. Hour deviation from user's typical active hour
    user_typical_hour = feat.groupby("user_id")["hour_of_day"].transform("mean")
    feat["hour_deviation"] = (feat["hour_of_day"] - user_typical_hour).abs()

    # 2. Amount ratio vs user mean
    feat["amount_ratio"] = feat["amount"] / feat["user_mean_amount"].clip(lower=1)

    # 3. Amount ratio vs user std (volatility)
    feat["amount_volatility"] = feat["amount"] / feat["user_std_amount"].clip(lower=1)

    # 4. Merchant risk score
    feat["merchant_risk"] = feat["merchant_category"].map(MERCHANT_RISK).fillna(0.3)

    # 5. UPI app risk score
    feat["app_risk"] = feat["upi_app"].map(UPI_APP_RISK).fillna(0.12)

    # 6. Capped time since last tx
    feat["time_since_last_tx_capped"] = feat["time_since_last_tx"].clip(upper=86400)

    # 7. Tx velocity interaction (2min × 5min)
    feat["velocity_interaction"] = feat["tx_count_2min"] * feat["tx_count_5min"]

    # 8. Is late night (1am-5am)
    feat["is_late_night"] = ((feat["hour_of_day"] >= 1) & (feat["hour_of_day"] <= 5)).astype(int)

    # 9. Is high amount (>10x user mean)
    feat["is_high_amount"] = (feat["amount_ratio"] > 10).astype(int)

    # 10. New device + high velocity interaction
    feat["new_device_velocity"] = feat["is_new_device"].astype(int) * feat["tx_count_2min"]

    # 11. New device + high amount
    feat["new_device_high_amt"] = feat["is_new_device"].astype(int) * feat["is_high_amount"]

    # 12. Amount Z-score clipped
    feat["amount_zscore_clipped"] = feat["amount_zscore"].clip(-5, 5)

    # ── NEW HIGH-DISCRIMINATING FEATURES ──────────────────
    # 13. Direct high-risk merchant flag (crypto + gambling)
    feat["is_high_risk_merchant"] = feat["merchant_category"].isin(
        ["crypto_exchange", "gambling"]
    ).astype(int)

    # 14. City anomaly — is current city different from user's most common city?
    user_home_city = feat.groupby("user_id")["location_city"].transform(
        lambda x: x.mode()[0] if len(x) > 0 else x.iloc[0]
    )
    feat["is_city_anomaly"] = (feat["location_city"] != user_home_city).astype(int)

    # 15. Exact burst flag — tx_count_2min >= 5
    feat["is_burst"] = (feat["tx_count_2min"] >= 5).astype(int)

    # 16. Late night + high amount combo
    feat["late_night_high_amt"] = feat["is_late_night"] * feat["is_high_amount"]

    # 17. Very recent tx (last tx < 10 seconds ago)
    feat["was_very_recent"] = (feat["time_since_last_tx"] < 10).astype(int)

    # 18. High velocity flag (2min count >= 3)
    feat["high_velocity_flag"] = (feat["tx_count_2min"] >= 3).astype(int)

    # ── CATEGORICAL ENCODING ──────────────────────────────
    if encoders is None:
        encoders = {}

    cat_cols = ["merchant_category", "upi_app", "transaction_type", "location_city"]
    for col in cat_cols:
        if fit_encoders:
            le = LabelEncoder()
            feat[f"{col}_enc"] = le.fit_transform(feat[col].astype(str))
            encoders[col] = le
        else:
            le = encoders.get(col)
            if le:
                known = set(le.classes_)
                feat[f"{col}_enc"] = feat[col].apply(
                    lambda x: le.transform([x])[0] if x in known else -1
                )
            else:
                feat[f"{col}_enc"] = 0

    # ── FINAL FEATURE SET ──────────────────────────────────
    FEATURE_COLS = [
        # Base schema features
        "amount", "hour_of_day", "is_new_device", "is_weekend", "day_of_week",
        "tx_count_2min", "tx_count_5min", "time_since_last_tx_capped",
        "user_mean_amount", "user_std_amount",
        # Engineered features (original 12)
        "amount_zscore", "amount_zscore_clipped", "hour_deviation",
        "amount_ratio", "amount_volatility", "merchant_risk", "app_risk",
        "velocity_interaction", "is_late_night", "is_high_amount",
        "new_device_velocity", "new_device_high_amt",
        # New high-discriminating features
        "is_high_risk_merchant", "is_city_anomaly", "is_burst",
        "late_night_high_amt", "was_very_recent", "high_velocity_flag",
        # Encoded categoricals
        "merchant_category_enc", "upi_app_enc", "transaction_type_enc", "location_city_enc",
    ]

    feat["is_new_device"] = feat["is_new_device"].astype(int)
    X = feat[FEATURE_COLS].fillna(0)
    return X, encoders, FEATURE_COLS


def train(df: pd.DataFrame) -> tuple:

    """
    Train XGBoost classifier with stratified 70/15/15 split.
    scale_pos_weight = 30.25 handles 3.2% fraud imbalance.
    """
    print("[XGB] Building feature matrix...")
    X, encoders, feature_cols = build_features(df, fit_encoders=True)
    y = df["fraud_label"].values

    # Save feature names
    with open(FEATURE_NAMES_PATH, "w") as f:
        json.dump(feature_cols, f)

    # 70% train, 15% val, 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )

    print(f"[XGB] Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    print(f"[XGB] Fraud in train: {y_train.sum():,} ({y_train.mean()*100:.2f}%)")

    # scale_pos_weight = (# negative samples) / (# positive samples)
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    spw = neg / pos
    print(f"[XGB] scale_pos_weight = {spw:.2f}")

    model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,
        use_label_encoder=False,
        eval_metric="aucpr",
        early_stopping_rounds=30,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    print("[XGB] Training XGBoost...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )

    # Save model and encoders
    joblib.dump(model, MODEL_PATH)
    joblib.dump(encoders, ENCODERS_PATH)
    print(f"[XGB] Model saved → {MODEL_PATH}")

    # Evaluate on test set
    metrics = evaluate_on_split(model, X_test, y_test, split_name="Test")

    # Save metrics
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2, cls=NumpyEncoder)

    with open(FEATURE_NAMES_PATH, "w") as f:
        json.dump(feature_cols, f)

    print(f"[XGB] Metrics saved → {METRICS_PATH}")
    return model, encoders, metrics


def evaluate_on_split(model: XGBClassifier, X, y, split_name: str = "Test", threshold: float = 0.4) -> dict:
    """
    Evaluate XGBoost on a data split. Uses threshold=0.4 to boost recall.
    """
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()

    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    auc = roc_auc_score(y, y_prob)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    metrics = {
        "split": split_name,
        "threshold": threshold,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4),
        "false_positive_rate": round(fpr, 4),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "total_samples": int(len(y)),
    }

    print(f"\n=== XGBoost {split_name} Set Metrics (threshold={threshold}) ===")
    print(f"  Precision  : {precision:.4f}  (target: ≥ 0.88)")
    print(f"  Recall     : {recall:.4f}  (target: ≥ 0.85)")
    print(f"  F1-Score   : {f1:.4f}  (target: ≥ 0.86)")
    print(f"  ROC-AUC    : {auc:.4f}  (target: ≥ 0.93)")
    print(f"  FPR        : {fpr:.4f}  (target: ≤ 0.02)")
    print(f"  TP: {tp}  FP: {fp}  TN: {tn}  FN: {fn}")

    return metrics


def load() -> tuple:
    """Load saved XGBoost model and encoders."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train() first.")
    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODERS_PATH) if os.path.exists(ENCODERS_PATH) else {}
    return model, encoders


def score_transaction(txn: dict, model: XGBClassifier = None, encoders: dict = None) -> dict:
    """
    Score a single transaction against the XGBoost model.

    Returns a dict with:
        risk_score          (float)  : raw fraud probability from XGBoost (0.0 – 1.0)
        should_display_alert (bool)  : True when risk_score >= DISPLAY_THRESHOLD (0.40)
                                       → transaction is flagged in the live feed
        should_block         (bool)  : True when risk_score >= BLOCK_THRESHOLD (0.90)
                                       → transaction is hard-blocked and triggers red alert
        risk_pts            (float)  : 0-35 pt contribution to the ensemble risk score
                                       (XGBoost weight = 35%)
    """
    if model is None or encoders is None:
        model, encoders = load()

    # Build a minimal dataframe for feature engineering
    txn_df = pd.DataFrame([txn])

    # Provide population-level defaults for missing engineered fields
    defaults = {
        "user_mean_amount": txn.get("user_mean_amount", 500),
        "user_std_amount": txn.get("user_std_amount", 300),
        "amount_zscore": txn.get("amount_zscore", 0.0),
        "tx_count_2min": txn.get("tx_count_2min", 0),
        "tx_count_5min": txn.get("tx_count_5min", 0),
        "time_since_last_tx": txn.get("time_since_last_tx", 99999),
        "day_of_week": txn.get("day_of_week", 0),
        "is_weekend": txn.get("is_weekend", 0),
        "hour_of_day": txn.get("hour_of_day", 12),
    }
    for k, v in defaults.items():
        if k not in txn_df.columns:
            txn_df[k] = v

    X, _, _ = build_features(txn_df, fit_encoders=False, encoders=encoders)
    fraud_prob = float(model.predict_proba(X)[0][1])

    return {
        "risk_score":           round(fraud_prob, 6),
        "should_display_alert": fraud_prob >= DISPLAY_THRESHOLD,
        "should_block":         fraud_prob >= BLOCK_THRESHOLD,
        "risk_pts":             round(fraud_prob * 35, 2),  # XGBoost's 35% ensemble contribution
    }


if __name__ == "__main__":
    print("Loading dataset...")
    df = pd.read_csv("upi_transactions.csv")
    model, encoders, metrics = train(df)
    print("\n=== TRAINING COMPLETE ===")
    print(f"F1: {metrics['f1_score']} | AUC: {metrics['roc_auc']} | FPR: {metrics['false_positive_rate']}")
