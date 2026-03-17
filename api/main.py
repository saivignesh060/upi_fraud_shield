from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import asyncio
import os
from collections import deque

from models.xgboost_classifier import build_features, load as load_xgb, DISPLAY_THRESHOLD, BLOCK_THRESHOLD, score_transaction
from explainability.shap_explainer import explain_transaction, load_explainer, get_shap_summary
from explainability.nl_explanation import generate_explanation
from models.behavioral_profiler import UserProfileEngine

app = FastAPI(title="UPI FraudShield API - M4 Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Application State
xgb_model = None
xgb_encoders = None
shap_explainer_model = None
profile_engine = None

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
    "fpr": 0.0
}

@app.on_event("startup")
async def startup_event():
    global xgb_model, xgb_encoders, shap_explainer_model, profile_engine
    print("[API] Loading XGBoost Model & Encoders...")
    xgb_model, xgb_encoders = load_xgb()
    
    print("[API] Loading SHAP Explainer...")
    shap_explainer_model = load_explainer()
    
    print("[API] Loading User Profiles...")
    profile_engine = UserProfileEngine()
    profile_engine.load()
    print("[API] Startup Complete.")

@app.post("/score")
async def score_endpoint(request: Request):
    """
    Score a single raw transaction dictionary via REST POST
    """
    txn_dict = await request.json()
    return score_single_tx(txn_dict)

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
        stream_metrics["f1"] = round(2 * (stream_metrics["precision"] * stream_metrics["recall"]) / (stream_metrics["precision"] + stream_metrics["recall"]), 4)
    if (fp + tn) > 0:
        stream_metrics["fpr"] = round(fp / (fp + tn), 4)

def score_single_tx(txn_dict: dict) -> dict:
    """
    Internal scoring logic used by both REST and WebSocket endpoints.
    Combines behavioral profiling, feature building, XGBoost scoring, and SHAP.
    """
    global stream_metrics
    user_id = txn_dict.get("user_id", "unknown")
    
    # 1. Get behavioral profile (handles cold-start via population defaults internally)
    user_profile = profile_engine.get_profile(user_id)
    
    # 2. Inject profile fields into transaction for feature building
    txn_enriched = txn_dict.copy()
    txn_enriched["user_mean_amount"] = user_profile.get("user_mean_amount", 500.0)
    txn_enriched["user_std_amount"] = user_profile.get("user_std_amount", 300.0)
    
    # Pre-calculate amount_zscore based on live profile metrics
    amt = float(txn_enriched.get("amount", 0.0))
    std = txn_enriched["user_std_amount"]
    if std > 0:
        txn_enriched["amount_zscore"] = (amt - txn_enriched["user_mean_amount"]) / std
    else:
        txn_enriched["amount_zscore"] = 0.0
        
    # Inject defaults for engineered features if missing
    defaults = {
        "tx_count_2min": txn_enriched.get("tx_count_2min", 0),
        "tx_count_5min": txn_enriched.get("tx_count_5min", 0),
        "time_since_last_tx": txn_enriched.get("time_since_last_tx", 99999),
        "day_of_week": txn_enriched.get("day_of_week", 0),
        "is_weekend": txn_enriched.get("is_weekend", 0),
        "hour_of_day": txn_enriched.get("hour_of_day", 12),
        "is_new_device": txn_enriched.get("is_new_device", False)
    }
    for k, v in defaults.items():
        if k not in txn_enriched:
            txn_enriched[k] = v

    # 3. Create DataFrame and build the 32 required XGBoost features
    txn_df = pd.DataFrame([txn_enriched])
    X_row, _, _ = build_features(txn_df, fit_encoders=False, encoders=xgb_encoders)
    
    # 4. Score with XGBoost model
    fraud_prob = float(xgb_model.predict_proba(X_row)[0][1])
    should_display = fraud_prob >= DISPLAY_THRESHOLD
    should_block = fraud_prob >= BLOCK_THRESHOLD
    
    # Metrics Tracking
    predicted_label = 1 if should_display else 0
    actual_label = int(txn_dict.get("fraud_label", 0))
    
    stream_metrics["total_scored"] += 1
    if actual_label == 1 and predicted_label == 1:
        stream_metrics["tp"] += 1
    elif actual_label == 0 and predicted_label == 1:
        stream_metrics["fp"] += 1
    elif actual_label == 1 and predicted_label == 0:
        stream_metrics["fn"] += 1
    elif actual_label == 0 and predicted_label == 0:
        stream_metrics["tn"] += 1
        
    calc_metrics()
    
    # 5. Extract explainability if flagged
    shap_top3 = []
    nl_exp = {}
    fraud_type_predicted = txn_dict.get("fraud_type", "legitimate")
    
    if should_display:
        shap_res = explain_transaction(X_row, explainer=shap_explainer_model, top_k=3)
        shap_top3 = get_shap_summary(shap_res)["top_features"]
        
        # Heuristic to map top feature to fraud type for demonstration
        top_name = shap_top3[0]["name"].lower() if shap_top3 else ""
        if "velocity" in top_name or "count" in top_name or "burst" in top_name:
            fraud_type_predicted = "burst"
        elif "amount" in top_name or "ratio" in top_name:
            fraud_type_predicted = "amount_spike"
        elif "location" in top_name or "city" in top_name:
            fraud_type_predicted = "location_jump"
        elif "night" in top_name or "hour" in top_name or "time" in top_name:
            fraud_type_predicted = "night_anomaly"
        elif "device" in top_name:
            fraud_type_predicted = "device_switch"
        elif "merchant" in top_name:
            fraud_type_predicted = "merchant_anomaly"
        else:
            fraud_type_predicted = "anomaly"
            
        nl_exp = generate_explanation(
            fraud_type=fraud_type_predicted,
            transaction=txn_enriched,
            user_profile=user_profile,
            shap_features=shap_res if should_display else [],
            risk_score=float(fraud_prob * 100)
        )
        
        # Deduplicate alerts
        tx_id = txn_dict.get("transaction_id", "N/A")
        existing_ids = {a["transaction_id"] for a in alerts_list}
        
        if tx_id not in existing_ids:
            # Save to alerts feed
            alerts_list.appendleft({
                "transaction_id": tx_id,
                "user_id": user_id,
                "amount": amt,
                "merchant_category": txn_dict.get("merchant_category", "unknown"),
                "timestamp": txn_dict.get("timestamp", ""),
                "risk_score": round(fraud_prob, 4),
                "should_block": should_block,
                "fraud_type_predicted": fraud_type_predicted,
                "true_fraud_type": txn_dict.get("fraud_type", "legitimate"),
                "shap": shap_top3,
                "explanation": nl_exp,
            })
        
    # 6. Update local user profile for future tracking
    profile_engine.update_profile(user_id, txn_dict)

    return {
        "risk_score": round(fraud_prob, 6),
        "should_display_alert": should_display,
        "should_block": should_block,
        "fraud_type_predicted": fraud_type_predicted,
        "shap_top3": shap_top3,
        "explanation": nl_exp
    }


@app.websocket("/stream")
async def websocket_stream(websocket: WebSocket, tps: int = 20, rows: int = 10000):
    """
    WebSocket endpoint that replays the CSV Dataset at adjustable speed.
    """
    await websocket.accept()
    
    csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "upi_transactions.csv")
    try:
        df = pd.read_csv(csv_path)
        if rows > 0:
            df = df.head(rows)
    except Exception as e:
        await websocket.send_text(f"Error loading dataset: {e}")
        await websocket.close()
        return
        
    delay = 1.0 / tps if tps > 0 else 0.2
    
    for row in df.itertuples(index=False):
        txn_dict = row._asdict()
        try:
            # Score transaction live
            res = score_single_tx(txn_dict)
            
            # Combine input dict with scoring payload
            out_payload = {**txn_dict, **res}
            await websocket.send_json(out_payload)
            
            await asyncio.sleep(delay)
        except WebSocketDisconnect:
            print("[API] Stream client disconnected")
            break
        except Exception as e:
            print(f"[API] Error processing transaction: {e}")
            await asyncio.sleep(delay)


@app.get("/profile/{user_id}")
async def get_user_profile(user_id: str):
    """Retrieves behavioral profile variables for a given user"""
    prof = profile_engine.get_profile(user_id)
    # Convert sets to lists for JSON serialization
    safe_prof = prof.copy()
    safe_prof["known_devices"] = list(safe_prof["known_devices"])
    safe_prof["known_cities"] = list(safe_prof["known_cities"])
    safe_prof["preferred_merchant_categories"] = list(safe_prof["preferred_merchant_categories"])
    return safe_prof


@app.get("/alerts")
async def get_alerts():
    """Returns the last 100 flagged transactions from memory"""
    return list(alerts_list)


@app.get("/metrics")
async def get_metrics():
    """Returns live accuracy, precision, recall, F1 computation"""
    return stream_metrics

if __name__ == "__main__":
    import uvicorn
    # Make sure to run the API at port 8000 for standard local testing
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
