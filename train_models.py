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

from models.isolation_forest import train as train_if, evaluate as eval_if, build_features as if_features
from models.xgboost_classifier import train as train_xgb, build_features as xgb_features
from explainability.shap_explainer import build_explainer

DATASET_PATH = "upi_transactions.csv"
REPORT_PATH = "models/evaluation_report.json"

def run():
    print("=" * 60)
    print("  UPI FraudShield — M2 Model Training Pipeline")
    print("=" * 60)
    start_total = time.time()

    # ── LOAD DATA ─────────────────────────────────────────────
    print("\n[Step 1/5] Loading dataset...")
    t0 = time.time()
    df = pd.read_csv(DATASET_PATH)
    print(f"  Loaded {len(df):,} rows in {time.time()-t0:.1f}s")
    print(f"  Fraud: {df['fraud_label'].sum():,} ({df['fraud_label'].mean()*100:.2f}%)")

    # ── ISOLATION FOREST ──────────────────────────────────────
    print("\n[Step 2/5] Training Isolation Forest (unsupervised)...")
    t0 = time.time()
    if_model = train_if(df, contamination=0.032)
    if_metrics = eval_if(df, if_model)
    print(f"  Done in {time.time()-t0:.1f}s")
    print(f"  IF Recall: {if_metrics['recall']:.4f} | Precision: {if_metrics['precision']:.4f}")

    # ── XGBOOST ───────────────────────────────────────────────
    print("\n[Step 3/5] Training XGBoost classifier (supervised)...")
    t0 = time.time()
    xgb_model, encoders, xgb_metrics = train_xgb(df)
    print(f"  Done in {time.time()-t0:.1f}s")

    # ── SHAP EXPLAINER ────────────────────────────────────────
    print("\n[Step 4/5] Building SHAP TreeExplainer...")
    t0 = time.time()
    X_sample, _, _ = xgb_features(df.head(2000), fit_encoders=False, encoders=encoders)
    shap_explainer = build_explainer(xgb_model, X_sample)
    print(f"  Done in {time.time()-t0:.1f}s")

    # ── EVALUATION REPORT ─────────────────────────────────────
    print("\n[Step 5/5] Generating evaluation report...")
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
    print(f"    Precision  : {xgb_metrics['precision']:.4f}  {'✅' if xgb_metrics['precision'] >= 0.88 else '⚠️ below target'}")
    print(f"    Recall     : {xgb_metrics['recall']:.4f}  {'✅' if xgb_metrics['recall'] >= 0.85 else '⚠️ below target'}")
    print(f"    F1-Score   : {xgb_metrics['f1_score']:.4f}  {'✅' if xgb_metrics['f1_score'] >= 0.86 else '⚠️ below target'}")
    print(f"    ROC-AUC    : {xgb_metrics['roc_auc']:.4f}  {'✅' if xgb_metrics['roc_auc'] >= 0.93 else '⚠️ below target'}")
    print(f"    FPR        : {xgb_metrics['false_positive_rate']:.4f}  {'✅' if xgb_metrics['false_positive_rate'] <= 0.02 else '⚠️ above target'}")
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
