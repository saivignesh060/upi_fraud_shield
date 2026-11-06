"""
Real-world fraud benchmark (separate from the UPI synthetic pipeline).

Default dataset: ULB Credit Card Fraud (`benchmarks/creditcard.csv`).
Optional IEEE-CIS: place `train_transaction.csv` (+ optional `train_identity.csv`)
under `benchmarks/ieee_cis/` and pass --dataset ieee.

Split: ordered by Time / TransactionDT — first 70% train, next 15% val, last 15% test.
Encoders/aggregates fit on train only. XGBoost hyperparameters match models/xgboost_classifier.py.
Threshold chosen on validation (max recall s.t. precision >= 0.5).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.xgboost_classifier import select_threshold  # noqa: E402

RESULTS_JSON = Path(__file__).resolve().parent / "results.json"
RESULTS_MD = Path(__file__).resolve().parent / "RESULTS.md"


def load_ulb(path: Path) -> tuple[pd.DataFrame, str]:
    df = pd.read_csv(path)
    if "Class" not in df.columns or "Time" not in df.columns:
        raise ValueError(f"ULB file missing Class/Time columns: {path}")
    df = df.sort_values("Time").reset_index(drop=True)
    y = df["Class"].astype(int)
    X = df.drop(columns=["Class"])
    return X, y, "ULB Credit Card Fraud (creditcard.csv)"


def load_ieee(dir_path: Path) -> tuple[pd.DataFrame, str]:
    txn_path = dir_path / "train_transaction.csv"
    if not txn_path.exists():
        raise FileNotFoundError(txn_path)
    df = pd.read_csv(txn_path)
    id_path = dir_path / "train_identity.csv"
    if id_path.exists():
        ident = pd.read_csv(id_path)
        df = df.merge(ident, on="TransactionID", how="left")
    df = df.sort_values("TransactionDT").reset_index(drop=True)
    y = df["isFraud"].astype(int)
    drop_cols = [c for c in ("isFraud", "TransactionID") if c in df.columns]
    X = df.drop(columns=drop_cols)
    # Encode object columns using train-only later; for now keep as object
    return X, y, "IEEE-CIS Fraud Detection"


def time_split(X: pd.DataFrame, y: pd.Series, train_frac=0.70, val_frac=0.15):
    n = len(X)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    n_test = n - n_train - n_val
    idx_train = slice(0, n_train)
    idx_val = slice(n_train, n_train + n_val)
    idx_test = slice(n_train + n_val, n)
    return (
        X.iloc[idx_train].copy(), y.iloc[idx_train].copy(),
        X.iloc[idx_val].copy(), y.iloc[idx_val].copy(),
        X.iloc[idx_test].copy(), y.iloc[idx_test].copy(),
        {"train": n_train, "val": n_val, "test": n_test},
    )


def encode_fit_transform(X_train, X_val, X_test):
    """Label-encode object columns using train classes only; unknown -> -1."""
    X_train, X_val, X_test = X_train.copy(), X_val.copy(), X_test.copy()
    encoders = {}
    for col in list(X_train.columns):
        if X_train[col].dtype == object or str(X_train[col].dtype) == "string":
            le = LabelEncoder()
            vals = X_train[col].astype(str).fillna("__NA__")
            X_train[col] = le.fit_transform(vals)
            encoders[col] = le
            known = set(le.classes_)

            def _map(series, le=le, known=known):
                return series.astype(str).fillna("__NA__").map(
                    lambda x: int(le.transform([x])[0]) if x in known else -1
                )

            X_val[col] = _map(X_val[col])
            X_test[col] = _map(X_test[col])
        else:
            # Fill NA with train median
            med = X_train[col].median()
            X_train[col] = X_train[col].fillna(med)
            X_val[col] = X_val[col].fillna(med)
            X_test[col] = X_test[col].fillna(med)
    return X_train, X_val, X_test, encoders


def evaluate(model, X, y, threshold, split_name):
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {
        "split": split_name,
        "threshold": float(threshold),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, prob)),
        "pr_auc": float(average_precision_score(y, prob)),
        "fraud_rate": float(y.mean()),
        "n": int(len(y)),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def run(dataset: str = "ulb"):
    if dataset == "ulb":
        path = ROOT / "benchmarks" / "creditcard.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Download ULB creditcard.csv to that path.\n"
                "Kaggle: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
                "Or: place creditcard.csv at benchmarks/creditcard.csv"
            )
        X, y, name = load_ulb(path)
    elif dataset == "ieee":
        d = ROOT / "benchmarks" / "ieee_cis"
        X, y, name = load_ieee(d)
    else:
        raise ValueError(dataset)

    print(f"Dataset: {name}")
    print(f"Rows: {len(X):,}  fraud_rate={y.mean()*100:.4f}%")

    X_train, y_train, X_val, y_val, X_test, y_test, sizes = time_split(X, y)
    print(f"Split sizes: {sizes}")
    print(
        f"Fraud rates — train {y_train.mean()*100:.4f}% | "
        f"val {y_val.mean()*100:.4f}% | test {y_test.mean()*100:.4f}%"
    )

    X_train, X_val, X_test, _ = encode_fit_transform(X_train, X_val, X_test)

    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    spw = neg / max(pos, 1)
    print(f"scale_pos_weight={spw:.2f}")

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
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)

    val_prob = model.predict_proba(X_val)[:, 1]
    thr_info = select_threshold(y_val.to_numpy(), val_prob)
    thr = float(thr_info["threshold"])
    print(f"Selected threshold={thr:.6f} via {thr_info['method']}")

    val_m = evaluate(model, X_val, y_val, thr, "Val")
    test_m = evaluate(model, X_test, y_test, thr, "Test")

    payload = {
        "dataset": name,
        "split": "time-ordered 70/15/15",
        "sizes": sizes,
        "fraud_rates": {
            "train": float(y_train.mean()),
            "val": float(y_val.mean()),
            "test": float(y_test.mean()),
        },
        "threshold_selection": thr_info,
        "val": val_m,
        "test": test_m,
        "note": "Separate from UPI synthetic pipeline; does not modify UPI models.",
    }
    RESULTS_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md = f"""# Real-data benchmark results

**Dataset:** {name}  
**Split:** time-ordered 70% train / 15% val / 15% test  
**Note:** Separate from the UPI synthetic dashboard pipeline.

## Fraud rates

| Split | N | Fraud rate |
|---|---:|---:|
| Train | {sizes['train']:,} | {y_train.mean()*100:.4f}% |
| Val | {sizes['val']:,} | {y_val.mean()*100:.4f}% |
| Test | {sizes['test']:,} | {y_test.mean()*100:.4f}% |

## Threshold (validation)

- Method: `{thr_info['method']}`
- Threshold: **{thr:.6f}**

## Test metrics

| Metric | Value |
|---|---:|
| ROC-AUC | {test_m['roc_auc']:.4f} |
| PR-AUC | {test_m['pr_auc']:.4f} |
| Precision | {test_m['precision']:.4f} |
| Recall | {test_m['recall']:.4f} |
| F1 | {test_m['f1']:.4f} |
"""
    RESULTS_MD.write_text(md, encoding="utf-8")
    print(md)
    return payload


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["ulb", "ieee"], default="ulb")
    args = ap.parse_args()
    run(args.dataset)
