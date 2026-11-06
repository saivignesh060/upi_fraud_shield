"""Profile one scored transaction end-to-end (feature build, predict, SHAP, JSON)."""
import json
import os
import sys
import time

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models.xgboost_classifier import (
    load as load_xgb,
    enrich_txn_from_profile,
    build_features,
    _load_display_threshold,
)
from models.behavioral_profiler import UserProfileEngine
from explainability.shap_explainer import load_explainer, explain_transaction, get_shap_summary


def main():
    model, encoders = load_xgb()
    explainer = load_explainer()
    engine = UserProfileEngine()
    engine.load()
    thr = _load_display_threshold()

    df = pd.read_csv(os.path.join(ROOT, "upi_transactions.csv"), nrows=5000)
    # Prefer a row likely to flag
    candidates = df[df["tx_count_2min"] >= 3] if "tx_count_2min" in df.columns else df
    row = candidates.iloc[0] if len(candidates) else df.iloc[0]
    txn = row.to_dict()
    for k, v in list(txn.items()):
        if hasattr(v, "item"):
            txn[k] = v.item()
        elif not isinstance(v, (str, int, float, bool)) and v is not None:
            txn[k] = str(v)

    timings = {}

    t0 = time.perf_counter()
    profile = engine.get_profile(txn["user_id"])
    enriched = enrich_txn_from_profile(txn, profile)
    timings["enrich_profile_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    X_row, _, _ = build_features(pd.DataFrame([enriched]), fit_encoders=False, encoders=encoders)
    timings["feature_build_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    prob = float(model.predict_proba(X_row)[0][1])
    timings["predict_ms"] = (time.perf_counter() - t0) * 1000

    flagged = prob >= thr
    timings["flagged"] = flagged
    timings["risk_score"] = round(prob, 6)
    timings["threshold"] = thr

    if flagged:
        t0 = time.perf_counter()
        shap_res = explain_transaction(X_row, explainer=explainer, top_k=3)
        shap_top3 = get_shap_summary(shap_res)["top_features"]
        timings["shap_ms"] = (time.perf_counter() - t0) * 1000
    else:
        shap_top3 = []
        timings["shap_ms"] = 0.0

    payload = {**txn, "risk_score": round(prob, 6), "shap_top3": shap_top3}
    t0 = time.perf_counter()
    raw = json.dumps(payload, default=str)
    timings["json_serialize_ms"] = (time.perf_counter() - t0) * 1000
    timings["json_bytes"] = len(raw.encode("utf-8"))

    # Simulated stream sleep for tps=20
    timings["sleep_budget_tps20_ms"] = 1000.0 / 20

    total_compute = (
        timings["enrich_profile_ms"]
        + timings["feature_build_ms"]
        + timings["predict_ms"]
        + timings["shap_ms"]
        + timings["json_serialize_ms"]
    )
    timings["total_compute_ms"] = total_compute

    print("=== STAGE BREAKDOWN (one transaction) ===")
    for k in (
        "enrich_profile_ms",
        "feature_build_ms",
        "predict_ms",
        "shap_ms",
        "json_serialize_ms",
        "total_compute_ms",
        "sleep_budget_tps20_ms",
    ):
        print(f"  {k}: {timings[k]:.3f}")
    print(f"  flagged: {flagged} risk={timings['risk_score']} thr={thr}")
    print(f"  slowest: {max((('feature_build', timings['feature_build_ms']), ('predict', timings['predict_ms']), ('shap', timings['shap_ms']), ('json', timings['json_serialize_ms'])), key=lambda x: x[1])}")


if __name__ == "__main__":
    main()
