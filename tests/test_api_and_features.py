"""Pytest coverage for feature building, /score, /metrics, and score/stream parity."""
import copy
import json
import os
import sys

import pandas as pd
import pytest
from fastapi.testclient import TestClient

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models.xgboost_classifier import (
    FEATURE_COLS,
    apply_user_baselines,
    build_features,
    compute_user_baselines,
    load as load_xgb,
    load_baselines,
    time_based_user_split,
    enrich_txn_from_profile,
)
import api.main as api_main


@pytest.fixture(scope="module")
def sample_df():
    csv_path = os.path.join(ROOT, "upi_transactions.csv")
    if not os.path.exists(csv_path):
        pytest.skip("upi_transactions.csv missing")
    return pd.read_csv(csv_path, nrows=5000)


def test_feature_count_is_32(sample_df):
    train_df, _, _ = time_based_user_split(sample_df)
    baselines = compute_user_baselines(train_df)
    fe = apply_user_baselines(train_df.head(20), baselines)
    X, enc, cols = build_features(fe, fit_encoders=True)
    assert len(cols) == 32
    assert list(X.columns) == FEATURE_COLS
    assert X.shape[1] == 32


def test_baselines_exclude_fraud_amounts(sample_df):
    train_df, _, _ = time_based_user_split(sample_df)
    baselines = compute_user_baselines(train_df)
    # Spot-check: for a user with both fraud and legit in train, mean matches legit-only
    for uid, grp in train_df.groupby("user_id"):
        legit = grp[grp["fraud_label"] == 0]
        if len(legit) < 2 or (grp["fraud_label"] == 1).sum() == 0:
            continue
        b = baselines[uid]
        expected = float(legit["amount"].mean())
        assert abs(b["user_mean_amount"] - expected) < 1e-6
        break
    else:
        pytest.skip("no mixed user in sample")


@pytest.fixture(scope="module")
def client():
    # Ensure artifacts exist
    if not os.path.exists(os.path.join(ROOT, "models", "xgboost_classifier.pkl")):
        pytest.skip("model artifacts missing")
    with TestClient(api_main.app) as c:
        yield c


def test_metrics_endpoint_shape(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    body = r.json()
    for key in ("total_scored", "tp", "fp", "tn", "fn", "precision", "recall", "f1", "fpr"):
        assert key in body


def test_score_endpoint_returns_risk(client, sample_df):
    txn = sample_df.iloc[0].to_dict()
    for k, v in list(txn.items()):
        if hasattr(v, "item"):
            txn[k] = v.item()
        elif not isinstance(v, (str, int, float, bool)) and v is not None:
            txn[k] = str(v)
    r = client.post("/score", json=txn)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["risk_score"] <= 1.0
    assert "should_display_alert" in body
    assert "display_threshold" in body


def test_score_stream_parity_same_profile_state(client, sample_df):
    """
    /score and the shared score_single_tx path (used by /stream) must agree
    when profile state is identical.
    """
    # Prefer a burst-like row if present in the head sample
    if "tx_count_2min" in sample_df.columns:
        candidates = sample_df[sample_df["tx_count_2min"] >= 3]
        row = candidates.iloc[0] if len(candidates) else sample_df.iloc[10]
    else:
        row = sample_df.iloc[10]
    txn = row.to_dict()
    for k, v in list(txn.items()):
        if hasattr(v, "item"):
            txn[k] = v.item()
        elif not isinstance(v, (str, int, float, bool)) and v is not None:
            txn[k] = str(v)

    uid = txn["user_id"]
    # Snapshot profile, score via REST, restore, score again via shared coroutine path
    existing = api_main.profile_engine.profiles.get(uid)
    prof_before = copy.deepcopy(existing) if existing is not None else None

    r1 = client.post("/score", json=txn)
    assert r1.status_code == 200
    score1 = r1.json()["risk_score"]

    # Restore profile to pre-score state (update_profile mutates)
    if prof_before is None:
        api_main.profile_engine.profiles.pop(uid, None)
    else:
        api_main.profile_engine.profiles[uid] = copy.deepcopy(prof_before)

    import asyncio
    score2_body = asyncio.run(api_main.score_single_tx(dict(txn)))
    score2 = score2_body["risk_score"]

    assert score1 == score2, f"parity failed: /score={score1} stream-path={score2}"
