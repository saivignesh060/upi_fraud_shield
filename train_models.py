"""
UPI FraudShield — M2 Training Runner
Trains all M2 models: Isolation Forest, XGBoost, and builds SHAP explainer.
Run this script from the project root: python train_models.py

Output files:
  models/isolation_forest.pkl
  models/xgboost_classifier.pkl
  models/xgb_encoders.pkl
  models/xgb_feature_names.json
  models/xgb_metrics.json
  explainability/shap_explainer.pkl
"""

import sys
import os
import json
import time
import numpy as np
import pandas as pd


class NumpyEncoder(json.JSONEncoder):
    """Handle numpy types that aren't JSON serializable by default."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.isolation_forest import train as train_if, evaluate as eval_if
from models.xgboost_classifier import (
    train as train_xgb,
    build_features as xgb_features,
    time_based_user_split,
    apply_user_baselines,
    load_baselines,
)
from models.behavioral_profiler import build_profiles_from_dataframe
from explainability.shap_explainer import build_explainer

DATASET_PATH = "upi_transactions.csv"
REPORT_PATH = "models/evaluation_report.json"

def run():
    print("=" * 60)
    print("  UPI FraudShield — M2 Model Training Pipeline")
    print("=" * 60)
    start_total = time.time()

    # ── LOAD DATA ─────────────────────────────────────────────
    print("\n[Step 1/6] Loading dataset...")
    t0 = time.time()
    df = pd.read_csv(DATASET_PATH, low_memory=False)
    print(f"  Loaded {len(df):,} rows in {time.time()-t0:.1f}s")
    print(f"  Fraud: {df['fraud_label'].sum():,} ({df['fraud_label'].mean()*100:.2f}%)")

    # ── ISOLATION FOREST ──────────────────────────────────────
    print("\n[Step 2/6] Training Isolation Forest (unsupervised)...")
    t0 = time.time()
    if_model = train_if(df, contamination=0.032)
    if_metrics = eval_if(df, if_model)
    print(f"  Done in {time.time()-t0:.1f}s")
    print(f"  IF Recall: {if_metrics['recall']:.4f} | Precision: {if_metrics['precision']:.4f}")

    # ── XGBOOST (leakage-safe) ────────────────────────────────
    print("\n[Step 3/6] Training XGBoost classifier (time-split, train baselines)...")
    t0 = time.time()
    xgb_model, encoders, xgb_metrics = train_xgb(df)
    print(f"  Done in {time.time()-t0:.1f}s")

    # ── PROFILES from train legit only ────────────────────────
    print("\n[Step 4/6] Rebuilding user profiles from TRAIN legitimate rows...")
    t0 = time.time()
    train_df, _, _ = time_based_user_split(df)
    build_profiles_from_dataframe(train_df[train_df["fraud_label"] == 0])
    print(f"  Done in {time.time()-t0:.1f}s")

    # ── SHAP EXPLAINER (train background only) ────────────────
    print("\n[Step 5/6] Building SHAP TreeExplainer (train background)...")
    t0 = time.time()
    baselines = load_baselines()
    train_fe = apply_user_baselines(train_df, baselines)
    X_sample, _, _ = xgb_features(train_fe.head(2000), fit_encoders=False, encoders=encoders)
    shap_explainer = build_explainer(xgb_model, X_sample)
    print(f"  Done in {time.time()-t0:.1f}s")

    # ── EVALUATION REPORT ─────────────────────────────────────
    print("\n[Step 6/6] Generating evaluation report...")
    report = {
        "isolation_forest": if_metrics,
        "xgboost": xgb_metrics,
        "prd_targets": {
            "precision_target": 0.88,
            "recall_target": 0.85,
            "f1_target": 0.86,
            "roc_auc_target": 0.93,
            "fpr_target": 0.02,
        },
        "prd_compliance": {
            "precision_ok": xgb_metrics["precision"] >= 0.88,
            "recall_ok": xgb_metrics["recall"] >= 0.85,
            "f1_ok": xgb_metrics["f1_score"] >= 0.86,
            "roc_auc_ok": xgb_metrics["roc_auc"] >= 0.93,
            "fpr_ok": xgb_metrics["false_positive_rate"] <= 0.02,
        },
        "training_time_seconds": round(time.time() - start_total, 1),
    }

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2, cls=NumpyEncoder)

    # ── SUMMARY ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  M2 TRAINING COMPLETE")
    print("=" * 60)
    print(f"\n  XGBoost Performance (Test Set):")
    print(f"    Threshold  : {xgb_metrics['threshold']:.6f}")
    print(f"    Precision  : {xgb_metrics['precision']:.4f}")
    print(f"    Recall     : {xgb_metrics['recall']:.4f}")
    print(f"    F1-Score   : {xgb_metrics['f1_score']:.4f}")
    print(f"    ROC-AUC    : {xgb_metrics['roc_auc']:.4f}")
    print(f"    PR-AUC     : {xgb_metrics.get('pr_auc', float('nan')):.4f}")
    print(f"    FPR        : {xgb_metrics['false_positive_rate']:.4f}")
    print(f"\n  Isolation Forest:")
    print(f"    Recall     : {if_metrics['recall']:.4f}")
    print(f"    Precision  : {if_metrics['precision']:.4f}")
    print(f"\n  Models saved to ./models/")
    print(f"  SHAP explainer saved to ./explainability/")
    print(f"  Full report: {REPORT_PATH}")
    print(f"  Total time : {time.time() - start_total:.1f}s")
    print("=" * 60)

    return report


if __name__ == "__main__":
    run()
