from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import asyncio
import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

from models.xgboost_classifier import (
    build_features,
    load as load_xgb,
    BLOCK_THRESHOLD,
    enrich_txn_from_profile,
    _load_display_threshold,
    load_baselines,
    apply_user_baselines,
)
from explainability.shap_explainer import explain_transaction, load_explainer, get_shap_summary
from explainability.nl_explanation import generate_explanation, generate_mitigation
from models.behavioral_profiler import UserProfileEngine

app = FastAPI(title="UPI FraudShield API - M4 Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

xgb_model = None
xgb_encoders = None
shap_explainer_model = None
profile_engine = None
DISPLAY_THRESHOLD = 0.40
_executor = ThreadPoolExecutor(max_workers=2)
BATCH_SIZE = 256

# Preloaded stream cache (built on first /stream or at startup if CSV present)
_stream_cache = {
    "csv_path": None,
    "txns": None,       # list[dict]
    "X": None,          # DataFrame feature matrix aligned with txns
}

alerts_list = deque(maxlen=100)
stream_metrics = {
    "total_scored": 0,
    "tp": 0,
    "fp": 0,
    "tn": 0,
    "fn": 0,
    "accuracy": 0.0,
    "precision": 0.0,
    "recall": 0.0,
    "f1": 0.0,
    "fpr": 0.0,
}


@app.on_event("startup")
async def startup_event():
    global xgb_model, xgb_encoders, shap_explainer_model, profile_engine, DISPLAY_THRESHOLD
    print("[API] Loading XGBoost Model & Encoders...")
    xgb_model, xgb_encoders = load_xgb()
    DISPLAY_THRESHOLD = _load_display_threshold(DISPLAY_THRESHOLD)
    print(f"[API] DISPLAY_THRESHOLD={DISPLAY_THRESHOLD}")

    print("[API] Loading SHAP Explainer...")
    shap_explainer_model = load_explainer()

    print("[API] Loading User Profiles...")
    profile_engine = UserProfileEngine()
    profile_engine.load()

    csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "upi_transactions.csv")
    if os.path.exists(csv_path):
        print("[API] Precomputing stream feature matrix (first 10k rows)...")
        _prepare_stream_cache(csv_path, rows=10000)
        print(f"[API] Stream cache ready: {len(_stream_cache['txns']):,} rows, "
              f"X={_stream_cache['X'].shape}")
    print("[API] Startup Complete.")


def _prepare_stream_cache(csv_path: str, rows: int = 10000):
    """Load CSV once, apply train baselines, build feature matrix for batch scoring."""
    df = pd.read_csv(csv_path)
    if rows > 0:
        df = df.head(rows).copy()
    baselines = load_baselines()
    if baselines:
        df = apply_user_baselines(df, baselines)
    else:
        # Fall back to CSV columns; ensure typical hour / home city exist
        if "user_typical_hour" not in df.columns:
            df["user_typical_hour"] = df["hour_of_day"].astype(float)
        if "user_home_city" not in df.columns:
            df["user_home_city"] = df["location_city"].astype(str)

    X, _, _ = build_features(df, fit_encoders=False, encoders=xgb_encoders)
    txns = df.to_dict(orient="records")
    for t in txns:
        for k, v in list(t.items()):
            if hasattr(v, "item"):
                t[k] = v.item()
            elif not isinstance(v, (str, int, float, bool, type(None))):
                t[k] = str(v)

    _stream_cache["csv_path"] = csv_path
    _stream_cache["txns"] = txns
    _stream_cache["X"] = X
    _stream_cache["rows"] = rows


@app.post("/score")
async def score_endpoint(request: Request):
    txn_dict = await request.json()
    return await score_single_tx(txn_dict)


def calc_metrics():
    global stream_metrics
    tp = stream_metrics["tp"]
    fp = stream_metrics["fp"]
    tn = stream_metrics["tn"]
    fn = stream_metrics["fn"]
    total = stream_metrics["total_scored"]

    if total > 0:
        stream_metrics["accuracy"] = round((tp + tn) / total, 4)
    if (tp + fp) > 0:
        stream_metrics["precision"] = round(tp / (tp + fp), 4)
    if (tp + fn) > 0:
        stream_metrics["recall"] = round(tp / (tp + fn), 4)
    if (stream_metrics["precision"] + stream_metrics["recall"]) > 0:
        stream_metrics["f1"] = round(
            2 * (stream_metrics["precision"] * stream_metrics["recall"])
            / (stream_metrics["precision"] + stream_metrics["recall"]),
            4,
        )
    if (fp + tn) > 0:
        stream_metrics["fpr"] = round(fp / (fp + tn), 4)


def _predict_proba_row(X_row):
    return float(xgb_model.predict_proba(X_row)[0][1])


def _predict_proba_batch(X_batch):
    return xgb_model.predict_proba(X_batch)[:, 1]


def _shap_top3(X_row):
    shap_res = explain_transaction(X_row, explainer=shap_explainer_model, top_k=3)
    return shap_res, get_shap_summary(shap_res)["top_features"]


def _fraud_type_from_shap(shap_top3, fallback="anomaly"):
    top_name = shap_top3[0]["name"].lower() if shap_top3 else ""
    if "velocity" in top_name or "count" in top_name or "burst" in top_name:
        return "burst"
    if "amount" in top_name or "ratio" in top_name:
        return "amount_spike"
    if "location" in top_name or "city" in top_name:
        return "location_jump"
    if "night" in top_name or "hour" in top_name or "time" in top_name:
        return "night_anomaly"
    if "device" in top_name:
        return "device_switch"
    if "merchant" in top_name:
        return "merchant_anomaly"
    return fallback


def _update_confusion(fraud_prob: float, actual_label: int):
    global stream_metrics, DISPLAY_THRESHOLD
    predicted_label = 1 if fraud_prob >= DISPLAY_THRESHOLD else 0
    stream_metrics["total_scored"] += 1
    if actual_label == 1 and predicted_label == 1:
        stream_metrics["tp"] += 1
    elif actual_label == 0 and predicted_label == 1:
        stream_metrics["fp"] += 1
    elif actual_label == 1 and predicted_label == 0:
        stream_metrics["fn"] += 1
    else:
        stream_metrics["tn"] += 1
    calc_metrics()


async def _attach_explainability(txn_dict, txn_enriched, X_row, fraud_prob, user_profile):
    shap_res, shap_top3 = await asyncio.get_running_loop().run_in_executor(
        _executor, _shap_top3, X_row
    )
    fraud_type_predicted = _fraud_type_from_shap(shap_top3)
    nl_exp = generate_explanation(
        fraud_type=fraud_type_predicted,
        transaction=txn_enriched,
        user_profile=user_profile,
        shap_features=shap_res,
        risk_score=float(fraud_prob * 100),
    )
    mitigation = generate_mitigation(shap_top3, fraud_type_predicted)
    should_block = fraud_prob >= BLOCK_THRESHOLD
    tx_id = txn_dict.get("transaction_id", "N/A")
    existing_ids = {a["transaction_id"] for a in alerts_list}
    if tx_id not in existing_ids:
        alerts_list.appendleft({
            "transaction_id": tx_id,
            "user_id": txn_dict.get("user_id", "unknown"),
            "amount": float(txn_enriched.get("amount", 0.0)),
            "merchant_category": txn_dict.get("merchant_category", "unknown"),
            "timestamp": txn_dict.get("timestamp", ""),
            "risk_score": round(fraud_prob, 4),
            "should_block": should_block,
            "fraud_type_predicted": fraud_type_predicted,
            "true_fraud_type": txn_dict.get("fraud_type", "legitimate"),
            "shap": shap_top3,
            "explanation": nl_exp,
            "mitigation": mitigation,
        })
    return shap_top3, nl_exp, mitigation, fraud_type_predicted


async def score_single_tx(txn_dict: dict, profile: bool = True) -> dict:
    """Shared scoring for REST (and fallback path)."""
    global DISPLAY_THRESHOLD
    user_id = txn_dict.get("user_id", "unknown")
    DISPLAY_THRESHOLD = _load_display_threshold(DISPLAY_THRESHOLD)

    user_profile = profile_engine.get_profile(user_id)
    txn_enriched = enrich_txn_from_profile(txn_dict, user_profile)

    txn_df = pd.DataFrame([txn_enriched])
    X_row, _, _ = build_features(txn_df, fit_encoders=False, encoders=xgb_encoders)
    fraud_prob = _predict_proba_row(X_row)
    should_display = fraud_prob >= DISPLAY_THRESHOLD
    should_block = fraud_prob >= BLOCK_THRESHOLD

    _update_confusion(fraud_prob, int(txn_dict.get("fraud_label", 0)))

    shap_top3 = []
    nl_exp = {}
    mitigation = None
    fraud_type_predicted = txn_dict.get("fraud_type", "legitimate")

    if should_display:
        shap_top3, nl_exp, mitigation, fraud_type_predicted = await _attach_explainability(
            txn_dict, txn_enriched, X_row, fraud_prob, user_profile
        )

    if profile:
        profile_engine.update_profile(user_id, txn_dict)

    return {
        "risk_score": round(fraud_prob, 6),
        "should_display_alert": should_display,
        "should_block": should_block,
        "fraud_type_predicted": fraud_type_predicted,
        "shap_top3": shap_top3,
        "explanation": nl_exp,
        "mitigation": mitigation,
        "display_threshold": DISPLAY_THRESHOLD,
    }


@app.websocket("/stream")
async def websocket_stream(websocket: WebSocket, tps: int = 20, rows: int = 10000):
    """
    Batch-scored CSV replay with precomputed features and a monotonic-clock scheduler.
    """
    global DISPLAY_THRESHOLD
    await websocket.accept()
    DISPLAY_THRESHOLD = _load_display_threshold(DISPLAY_THRESHOLD)

    csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "upi_transactions.csv")
    if not os.path.exists(csv_path):
        await websocket.send_json({
            "error": f"Dataset not found at {csv_path}. Run: python generate_dataset.py"
        })
        await websocket.close(code=1011, reason="Dataset CSV missing")
        return

    try:
        if (
            _stream_cache["txns"] is None
            or _stream_cache.get("rows") != rows
            or _stream_cache.get("csv_path") != csv_path
        ):
            _prepare_stream_cache(csv_path, rows=rows)
        txns = _stream_cache["txns"]
        X_all = _stream_cache["X"]
    except Exception as e:
        await websocket.send_json({"error": f"Error loading dataset: {e}"})
        await websocket.close(code=1011, reason="Dataset load failed")
        return

    interval = 1.0 / tps if tps > 0 else 0.05
    n = len(txns)
    probs = np.empty(n, dtype=np.float64)

    # Batch predict once up front (or in chunks to bound memory)
    try:
        for start in range(0, n, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n)
            probs[start:end] = _predict_proba_batch(X_all.iloc[start:end])
    except Exception as e:
        await websocket.send_json({"error": f"Batch scoring failed: {e}"})
        await websocket.close(code=1011, reason="Batch scoring failed")
        return

    target_time = time.monotonic()
    loop = asyncio.get_running_loop()

    for i, txn_dict in enumerate(txns):
        try:
            fraud_prob = float(probs[i])
            should_display = fraud_prob >= DISPLAY_THRESHOLD
            should_block = fraud_prob >= BLOCK_THRESHOLD
            _update_confusion(fraud_prob, int(txn_dict.get("fraud_label", 0)))

            shap_top3 = []
            nl_exp = {}
            mitigation = None
            fraud_type_predicted = txn_dict.get("fraud_type", "legitimate")

            if should_display:
                X_row = X_all.iloc[[i]]
                user_profile = profile_engine.get_profile(txn_dict.get("user_id", "unknown"))
                shap_top3, nl_exp, mitigation, fraud_type_predicted = await _attach_explainability(
                    txn_dict, txn_dict, X_row, fraud_prob, user_profile
                )

            profile_engine.update_profile(txn_dict.get("user_id", "unknown"), txn_dict)

            out_payload = {
                **txn_dict,
                "risk_score": round(fraud_prob, 6),
                "should_display_alert": should_display,
                "should_block": should_block,
                "fraud_type_predicted": fraud_type_predicted,
                "shap_top3": shap_top3,
                "explanation": nl_exp,
                "mitigation": mitigation,
                "display_threshold": DISPLAY_THRESHOLD,
            }
            await websocket.send_json(out_payload)

            # Monotonic-clock scheduler: catch up without drifting
            target_time += interval
            now = time.monotonic()
            sleep_for = target_time - now
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
            else:
                # Behind schedule — yield briefly so the event loop stays healthy
                await asyncio.sleep(0)
        except WebSocketDisconnect:
            print("[API] Stream client disconnected")
            break
        except Exception as e:
            print(f"[API] Error processing transaction: {e}")
            await asyncio.sleep(0)


@app.get("/profile/{user_id}")
async def get_user_profile(user_id: str):
    prof = profile_engine.get_profile(user_id)
    safe_prof = prof.copy()
    safe_prof["known_devices"] = list(safe_prof["known_devices"])
    safe_prof["known_cities"] = list(safe_prof["known_cities"])
    safe_prof["preferred_merchant_categories"] = list(safe_prof["preferred_merchant_categories"])
    return safe_prof


@app.get("/alerts")
async def get_alerts():
    return list(alerts_list)


@app.get("/metrics")
async def get_metrics():
    return stream_metrics


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
