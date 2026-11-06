"""
UPI FraudShield — XGBoost Supervised Fraud Classifier (Layer 3)
Leakage-safe training: per-user time split BEFORE feature engineering;
baselines computed from train legitimate rows only.
"""

import numpy as np
import pandas as pd
import joblib
import os
import json
from xgboost import XGBClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
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
THRESHOLD_PATH = os.path.join(os.path.dirname(__file__), "xgb_threshold.json")
BASELINES_PATH = os.path.join(os.path.dirname(__file__), "user_baselines.pkl")

BLOCK_THRESHOLD = 0.90


def _load_display_threshold(default: float = 0.40) -> float:
    if os.path.exists(THRESHOLD_PATH):
        try:
            with open(THRESHOLD_PATH) as f:
                return float(json.load(f)["display_threshold"])
        except Exception:
            return default
    return default


DISPLAY_THRESHOLD = _load_display_threshold()

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

FEATURE_COLS = [
    "amount", "hour_of_day", "is_new_device", "is_weekend", "day_of_week",
    "tx_count_2min", "tx_count_5min", "time_since_last_tx_capped",
    "user_mean_amount", "user_std_amount",
    "amount_zscore", "amount_zscore_clipped", "hour_deviation",
    "amount_ratio", "amount_volatility", "merchant_risk", "app_risk",
    "velocity_interaction", "is_late_night", "is_high_amount",
    "new_device_velocity", "new_device_high_amt",
    "is_high_risk_merchant", "is_city_anomaly", "is_burst",
    "late_night_high_amt", "was_very_recent", "high_velocity_flag",
    "merchant_category_enc", "upi_app_enc", "transaction_type_enc", "location_city_enc",
]


def time_based_user_split(df: pd.DataFrame, train_frac: float = 0.70, val_frac: float = 0.15):
    """
    Per-user chronological split: earliest train_frac -> train,
    next val_frac -> val, remainder -> test.
    """
    work = df.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce")
    work = work.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    train_idx, val_idx, test_idx = [], [], []
    for _, grp in work.groupby("user_id", sort=False):
        n = len(grp)
        idx = grp.index.to_numpy()
        if n == 1:
            train_idx.append(idx[0])
            continue
        if n == 2:
            train_idx.append(idx[0])
            test_idx.append(idx[1])
            continue
        n_train = max(1, int(n * train_frac))
        n_val = max(1, int(n * val_frac))
        if n_train + n_val >= n:
            n_val = max(1, n - n_train - 1) if n_train < n - 1 else 0
        n_test = n - n_train - n_val
        if n_test <= 0:
            n_test = 1
            n_train = max(1, n - n_val - n_test)
        train_idx.extend(idx[:n_train].tolist())
        val_idx.extend(idx[n_train:n_train + n_val].tolist())
        test_idx.extend(idx[n_train + n_val:].tolist())

    return work.loc[train_idx].copy(), work.loc[val_idx].copy(), work.loc[test_idx].copy()


def compute_user_baselines(train_df: pd.DataFrame) -> dict:
    """
    Train-only baselines from LEGITIMATE rows only (fraud amounts excluded).
    Returns user_id -> {user_mean_amount, user_std_amount, user_typical_hour, user_home_city}.
    """
    legit = train_df[train_df["fraud_label"] == 0].copy()
    if legit.empty:
        legit = train_df.copy()

    baselines = {}
    for uid, grp in legit.groupby("user_id"):
        mean_amt = float(grp["amount"].mean())
        std_amt = float(grp["amount"].std(ddof=1)) if len(grp) > 1 else 1.0
        if not np.isfinite(std_amt) or std_amt <= 0:
            std_amt = 1.0
        typical_hour = float(grp["hour_of_day"].mean())
        home_city = grp["location_city"].mode()
        home_city = str(home_city.iloc[0]) if len(home_city) else str(grp["location_city"].iloc[0])
        baselines[uid] = {
            "user_mean_amount": mean_amt,
            "user_std_amount": std_amt,
            "user_typical_hour": typical_hour,
            "user_home_city": home_city,
        }

    # Population fallbacks for cold-start users
    baselines["__population__"] = {
        "user_mean_amount": float(legit["amount"].mean()) if len(legit) else 500.0,
        "user_std_amount": float(legit["amount"].std(ddof=1) or 300.0),
        "user_typical_hour": float(legit["hour_of_day"].mean()) if len(legit) else 12.0,
        "user_home_city": str(legit["location_city"].mode().iloc[0]) if len(legit) else "Mumbai",
    }
    return baselines


def apply_user_baselines(df: pd.DataFrame, baselines: dict) -> pd.DataFrame:
    """Apply train baselines to any split (overwrites CSV user_* amount stats)."""
    out = df.copy()
    pop = baselines.get("__population__", {
        "user_mean_amount": 500.0,
        "user_std_amount": 300.0,
        "user_typical_hour": 12.0,
        "user_home_city": "Mumbai",
    })

    means, stds, hours, cities = [], [], [], []
    for uid in out["user_id"]:
        b = baselines.get(uid, pop)
        means.append(b["user_mean_amount"])
        stds.append(b["user_std_amount"])
        hours.append(b["user_typical_hour"])
        cities.append(b["user_home_city"])

    out["user_mean_amount"] = means
    out["user_std_amount"] = stds
    out["user_typical_hour"] = hours
    out["user_home_city"] = cities
    out["amount_zscore"] = (out["amount"] - out["user_mean_amount"]) / out["user_std_amount"].clip(lower=1e-6)
    return out


def build_features(df: pd.DataFrame, fit_encoders: bool = False, encoders: dict = None) -> tuple:
    """
    Build the 32-feature matrix. Uses row-level user_typical_hour / user_home_city
    when present (no full-frame groupby leakage).
    """
    feat = df.copy()

    if "timestamp" in feat.columns:
        try:
            feat["timestamp"] = pd.to_datetime(feat["timestamp"], errors="coerce")
        except Exception:
            pass

    # Ensure baseline columns exist (API injects them; training applies them beforehand)
    if "user_mean_amount" not in feat.columns:
        feat["user_mean_amount"] = 500.0
    if "user_std_amount" not in feat.columns:
        feat["user_std_amount"] = 300.0
    if "amount_zscore" not in feat.columns:
        feat["amount_zscore"] = (feat["amount"] - feat["user_mean_amount"]) / feat["user_std_amount"].clip(lower=1e-6)

    if "user_typical_hour" not in feat.columns:
        # Safe fallback for single-row without profile: use own hour (deviation 0)
        feat["user_typical_hour"] = feat["hour_of_day"].astype(float)
    if "user_home_city" not in feat.columns:
        feat["user_home_city"] = feat["location_city"].astype(str)

    feat["hour_deviation"] = (feat["hour_of_day"].astype(float) - feat["user_typical_hour"].astype(float)).abs()
    feat["amount_ratio"] = feat["amount"] / feat["user_mean_amount"].clip(lower=1)
    feat["amount_volatility"] = feat["amount"] / feat["user_std_amount"].clip(lower=1)
    feat["merchant_risk"] = feat["merchant_category"].map(MERCHANT_RISK).fillna(0.3)
    feat["app_risk"] = feat["upi_app"].map(UPI_APP_RISK).fillna(0.12)
    feat["time_since_last_tx_capped"] = feat["time_since_last_tx"].clip(upper=86400)
    feat["velocity_interaction"] = feat["tx_count_2min"] * feat["tx_count_5min"]
    feat["is_late_night"] = ((feat["hour_of_day"] >= 1) & (feat["hour_of_day"] <= 5)).astype(int)
    feat["is_high_amount"] = (feat["amount_ratio"] > 10).astype(int)
    feat["new_device_velocity"] = feat["is_new_device"].astype(int) * feat["tx_count_2min"]
    feat["new_device_high_amt"] = feat["is_new_device"].astype(int) * feat["is_high_amount"]
    feat["amount_zscore_clipped"] = feat["amount_zscore"].clip(-5, 5)
    feat["is_high_risk_merchant"] = feat["merchant_category"].isin(
        ["crypto_exchange", "gambling"]
    ).astype(int)
    feat["is_city_anomaly"] = (feat["location_city"].astype(str) != feat["user_home_city"].astype(str)).astype(int)
    feat["is_burst"] = (feat["tx_count_2min"] >= 5).astype(int)
    feat["late_night_high_amt"] = feat["is_late_night"] * feat["is_high_amount"]
    feat["was_very_recent"] = (feat["time_since_last_tx"] < 10).astype(int)
    feat["high_velocity_flag"] = (feat["tx_count_2min"] >= 3).astype(int)

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
            if le is not None:
                known = set(le.classes_)
                feat[f"{col}_enc"] = feat[col].astype(str).apply(
                    lambda x: int(le.transform([x])[0]) if x in known else -1
                )
            else:
                feat[f"{col}_enc"] = 0

    feat["is_new_device"] = feat["is_new_device"].astype(int)
    X = feat[FEATURE_COLS].fillna(0)
    return X, encoders, FEATURE_COLS


def select_threshold(y_true, y_prob, min_precision: float = 0.5, fn_cost: float = 10.0, fp_cost: float = 1.0):
    """
    Choose threshold on validation:
    1) Maximize recall subject to precision >= min_precision
    2) Else minimize cost = fn_cost*FN + fp_cost*FP
    """
    candidates = np.unique(np.concatenate([
        np.linspace(0.01, 0.99, 99),
        y_prob,
    ]))
    candidates = np.sort(candidates)

    best_prec = None
    best_cost = None

    for thr in candidates:
        y_pred = (y_prob >= thr).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        cost = fn_cost * fn + fp_cost * fp

        if prec >= min_precision:
            if best_prec is None or rec > best_prec["recall"] or (
                rec == best_prec["recall"] and thr < best_prec["threshold"]
            ):
                best_prec = {
                    "threshold": float(thr),
                    "precision": float(prec),
                    "recall": float(rec),
                    "cost": float(cost),
                    "method": "max_recall_precision>=0.5",
                    "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                }

        if best_cost is None or cost < best_cost["cost"] or (
            cost == best_cost["cost"] and thr < best_cost["threshold"]
        ):
            best_cost = {
                "threshold": float(thr),
                "precision": float(prec),
                "recall": float(rec),
                "cost": float(cost),
                "method": "min_cost_fn10x",
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            }

    chosen = best_prec if best_prec is not None else best_cost
    return chosen


def evaluate_on_split(model: XGBClassifier, X, y, split_name: str = "Test", threshold: float = 0.4) -> dict:
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()

    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    auc = roc_auc_score(y, y_prob)
    pr_auc = average_precision_score(y, y_prob)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    metrics = {
        "split": split_name,
        "threshold": round(float(threshold), 6),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "false_positive_rate": round(float(fpr), 4),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "total_samples": int(len(y)),
    }

    print(f"\n=== XGBoost {split_name} Set Metrics (threshold={threshold:.4f}) ===")
    print(f"  Precision  : {precision:.4f}")
    print(f"  Recall     : {recall:.4f}")
    print(f"  F1-Score   : {f1:.4f}")
    print(f"  ROC-AUC    : {auc:.4f}")
    print(f"  PR-AUC     : {pr_auc:.4f}")
    print(f"  FPR        : {fpr:.4f}")
    print(f"  TP: {tp}  FP: {fp}  TN: {tn}  FN: {fn}")
    return metrics


def train(df: pd.DataFrame) -> tuple:
    """
    Leakage-safe train:
      1) per-user time split
      2) baselines from train legit only
      3) apply baselines, then build features
      4) fit encoders on train only
      5) select threshold on validation
      6) evaluate on test
    """
    global DISPLAY_THRESHOLD

    print("[XGB] Time-based per-user split (70/15/15) BEFORE feature engineering...")
    train_df, val_df, test_df = time_based_user_split(df)
    print(f"[XGB] Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
    print(f"[XGB] Fraud rates — train {train_df['fraud_label'].mean()*100:.2f}% | "
          f"val {val_df['fraud_label'].mean()*100:.2f}% | test {test_df['fraud_label'].mean()*100:.2f}%")

    print("[XGB] Computing user baselines from TRAIN legitimate rows only...")
    baselines = compute_user_baselines(train_df)
    joblib.dump(baselines, BASELINES_PATH)

    train_df = apply_user_baselines(train_df, baselines)
    val_df = apply_user_baselines(val_df, baselines)
    test_df = apply_user_baselines(test_df, baselines)

    print("[XGB] Building feature matrices (encoders fit on train only)...")
    X_train, encoders, feature_cols = build_features(train_df, fit_encoders=True)
    X_val, _, _ = build_features(val_df, fit_encoders=False, encoders=encoders)
    X_test, _, _ = build_features(test_df, fit_encoders=False, encoders=encoders)
    y_train = train_df["fraud_label"].astype(int).values.copy()
    y_val = val_df["fraud_label"].astype(int).values
    y_test = test_df["fraud_label"].astype(int).values

    # ~1% label noise on TRAIN only (val/test stay clean)
    TRAIN_LABEL_NOISE = 0.01
    rng = np.random.default_rng(42)
    noise_mask = rng.random(len(y_train)) < TRAIN_LABEL_NOISE
    y_train[noise_mask] = 1 - y_train[noise_mask]
    print(f"[XGB] Train label noise flipped {int(noise_mask.sum()):,} / {len(y_train):,} "
          f"({TRAIN_LABEL_NOISE*100:.1f}%)")

    with open(FEATURE_NAMES_PATH, "w") as f:
        json.dump(feature_cols, f)

    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    spw = neg / max(pos, 1)
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

    joblib.dump(model, MODEL_PATH)
    joblib.dump(encoders, ENCODERS_PATH)
    print(f"[XGB] Model saved → {MODEL_PATH}")

    print("[XGB] Selecting threshold on validation set...")
    y_val_prob = model.predict_proba(X_val)[:, 1]
    thr_info = select_threshold(y_val, y_val_prob)
    DISPLAY_THRESHOLD = float(thr_info["threshold"])
    with open(THRESHOLD_PATH, "w") as f:
        json.dump({
            "display_threshold": DISPLAY_THRESHOLD,
            "block_threshold": BLOCK_THRESHOLD,
            "selection": thr_info,
        }, f, indent=2, cls=NumpyEncoder)
    print(f"[XGB] Selected threshold={DISPLAY_THRESHOLD:.6f} via {thr_info['method']} "
          f"(val P={thr_info['precision']:.4f} R={thr_info['recall']:.4f})")

    val_metrics = evaluate_on_split(model, X_val, y_val, split_name="Val", threshold=DISPLAY_THRESHOLD)
    metrics = evaluate_on_split(model, X_test, y_test, split_name="Test", threshold=DISPLAY_THRESHOLD)
    metrics["val_metrics"] = val_metrics
    metrics["threshold_selection"] = thr_info

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2, cls=NumpyEncoder)
    print(f"[XGB] Metrics saved → {METRICS_PATH}")
    return model, encoders, metrics


def load() -> tuple:
    """Load saved XGBoost model and encoders; refresh DISPLAY_THRESHOLD from disk."""
    global DISPLAY_THRESHOLD
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train() first.")
    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODERS_PATH) if os.path.exists(ENCODERS_PATH) else {}
    DISPLAY_THRESHOLD = _load_display_threshold(DISPLAY_THRESHOLD)
    return model, encoders


def load_baselines() -> dict:
    if os.path.exists(BASELINES_PATH):
        return joblib.load(BASELINES_PATH)
    return {}


def enrich_txn_from_profile(txn: dict, user_profile: dict) -> dict:
    """Inject profile-derived fields so single-row scoring matches stream features."""
    enriched = dict(txn)
    mean_amt = float(user_profile.get("user_mean_amount", 500.0) or 500.0)
    std_amt = float(user_profile.get("user_std_amount", 300.0) or 300.0)
    if std_amt <= 0:
        std_amt = 300.0
    enriched["user_mean_amount"] = mean_amt
    enriched["user_std_amount"] = std_amt
    amt = float(enriched.get("amount", 0.0))
    enriched["amount_zscore"] = (amt - mean_amt) / std_amt

    typical_hours = user_profile.get("typical_hours") or {}
    if typical_hours:
        enriched["user_typical_hour"] = float(
            max(typical_hours.items(), key=lambda kv: kv[1])[0]
        )
    else:
        enriched["user_typical_hour"] = float(enriched.get("hour_of_day", 12))

    known_cities = user_profile.get("known_cities") or set()
    if isinstance(known_cities, (list, set)) and len(known_cities) > 0:
        # Prefer first known city as proxy for home when mode unavailable
        enriched["user_home_city"] = next(iter(known_cities))
    else:
        enriched["user_home_city"] = str(enriched.get("location_city", "Mumbai"))

    defaults = {
        "tx_count_2min": 0,
        "tx_count_5min": 0,
        "time_since_last_tx": 99999,
        "day_of_week": 0,
        "is_weekend": 0,
        "hour_of_day": 12,
        "is_new_device": False,
        "merchant_category": "unknown",
        "upi_app": "GPay",
        "transaction_type": "P2M",
        "location_city": enriched.get("user_home_city", "Mumbai"),
        "timestamp": "2024-01-01 12:00:00",
    }
    for k, v in defaults.items():
        if k not in enriched or enriched[k] is None:
            enriched[k] = v
    return enriched


def score_transaction(txn: dict, model: XGBClassifier = None, encoders: dict = None,
                      user_profile: dict = None) -> dict:
    if model is None or encoders is None:
        model, encoders = load()

    profile = user_profile or {
        "user_mean_amount": txn.get("user_mean_amount", 500),
        "user_std_amount": txn.get("user_std_amount", 300),
        "typical_hours": {},
        "known_cities": set(),
    }
    enriched = enrich_txn_from_profile(txn, profile)
    txn_df = pd.DataFrame([enriched])
    X, _, _ = build_features(txn_df, fit_encoders=False, encoders=encoders)
    fraud_prob = float(model.predict_proba(X)[0][1])
    thr = _load_display_threshold(DISPLAY_THRESHOLD)

    return {
        "risk_score": round(fraud_prob, 6),
        "should_display_alert": fraud_prob >= thr,
        "should_block": fraud_prob >= BLOCK_THRESHOLD,
        "risk_pts": round(fraud_prob * 35, 2),
    }


if __name__ == "__main__":
    print("Loading dataset...")
    df = pd.read_csv("upi_transactions.csv")
    model, encoders, metrics = train(df)
    print("\n=== TRAINING COMPLETE ===")
    print(f"F1: {metrics['f1_score']} | AUC: {metrics['roc_auc']} | PR-AUC: {metrics['pr_auc']} | thr: {metrics['threshold']}")
