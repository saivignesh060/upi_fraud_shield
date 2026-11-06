# UPI FraudShield — Verification Report (final)

**Date:** 2026-11-06  
**Method:** Commands executed; only measured numbers stated.

---

## STEP 1 — Repo state (measured)

| Fact | Measured |
|---|---|
| Remote `origin` | `https://github.com/saivignesh060/upi_fraud_shield.git` — `git push` succeeded (`bd2135c..05923d7`) |
| `upi_transactions.csv` | **500,000** rows; `fraud_label` **4,000** / **0.008** (0.80%) |
| Model features | **32** (`models/xgb_feature_names.json`) |
| REST endpoints | **4** (`POST /score`, `GET /profile/{user_id}`, `GET /alerts`, `GET /metrics`) |
| WebSocket routes | **1** (`WS /stream`) |
| Docs phrases | No README claim of Network Graph / D3 / coordinated attack / 20 tx/s. Bezier remains only as accurate GeoMap arc description. Metrics section states data is **synthetic**. |

---

## STEP 2 — Low-tps throughput anomaly

**Cause:** Stream scored a large window (batch of 256) before the first WebSocket send, so warm-up/batch fill dominated short runs at low `tps` (measured earlier: tps=20 → **14.20** msg/s vs tps=50 → **48.49** msg/s).

**Fix:** Interleave small adaptive predict batches (`STREAM_BATCH_SIZE=32`) with the monotonic scheduler; send each scored row on schedule instead of waiting for a full large batch.

**Re-measure** (`scripts/measure_throughput.py`, n=500, warmup=20):

| Requested tps | Achieved msg/s |
|---:|---:|
| 10 | 10.02 |
| 20 | 20.04 |
| 50 | 50.25 |
| 100 | 100.26 |

---

## STEP 3 — Real-data benchmark (separate from UPI)

**Dataset:** ULB Credit Card Fraud (`benchmarks/creditcard.csv`) — IEEE-CIS not used (download unavailable / not on disk).  
**Split:** time-ordered by `Time` — 70% train / 15% val / 15% test. Encoders/aggregates fit on train only.  
**Threshold:** val max recall s.t. precision ≥ 0.5 → **0.148434**.

| Split | N | Fraud rate |
|---|---:|---:|
| Train | 199,364 | 0.1926% |
| Val | 42,721 | 0.1311% |
| Test | 42,722 | 0.1217% |

| Test metric | Value |
|---|---:|
| ROC-AUC | 0.9823 |
| PR-AUC | 0.7647 |
| Precision | 0.4699 |
| Recall | 0.7500 |
| F1 | 0.5778 |

Source: `benchmarks/RESULTS.md`. Does **not** replace UPI synthetic models.

---

## STEP 4 — README polish

- Screenshots/GIF: `docs/dashboard_main.png` (102,869 B), `docs/dashboard_threat_intel.png` (23,343 B), `docs/live_feed.gif` (6,961,755 B) via Playwright.
- README sections present: Results (UPI + throughput + ULB), Limitations, Validation on a real-world dataset, Quick start.
- SHAP load: rebuild TreeExplainer from the XGBoost model if the pickle is incompatible with the active pandas.
- Clean-clone (local clone + fresh `.venv` + `pip install -r requirements.txt`):
  - `load()` → XGBClassifier, **32** features
  - `FRAUD_RATE` in generator = **0.008**
  - `GET /metrics` on uvicorn returned **200** after stream-cache warm-up (CSV copied for smoke; full `generate_dataset.py` not re-timed here)
  - `npm run build` in clean `dashboard/` (see Final)
- Blocked: `git clone https://github.com/...` hung with no progress in this environment; used `git clone --local` from the working tree instead.

---

## Synthetic UPI model metrics (latest train)

| Metric | Value |
|---|---:|
| Threshold | 0.794784 |
| ROC-AUC | 0.8901 |
| PR-AUC | 0.5572 |
| Precision | 0.4993 |
| Recall | 0.5434 |
| F1 | 0.5204 |
| Test N | 79,724 |

Hardening: `FRAUD_RATE=0.008`, hard negatives, ~70% signature strength, **1% train-only label noise**.

---

## Earlier profile (one scored transaction, before stream optimization)

| Stage | ms |
|---|---|
| enrich_profile | 0.061 |
| **feature_build (slowest)** | **172.569** |
| predict | 27.280 |
| shap (flagged) | 15.820 |
| json_serialize | 0.304 |
| total_compute | 216.035 |

---

## Feature checklist

| Item | Result |
|---|---|
| NetworkGraph removed | PASS — GeoMap-only TI; `d3` removed |
| Throughput anomaly fixed | PASS — rates in STEP 2 |
| Real-data benchmark | PASS — ULB results in STEP 3 |
| README + screenshots | PASS — STEP 4 |
| Harder synthetic data + retrain | PASS — metrics above |

---

## Limitations

- UPI metrics are on **synthetic** data, not production bank traffic.
- Harder mode still shares users/merchants across time splits; real fraud is noisier.
- Train label noise (1%) is artificial; test labels stay clean.
- Stream CSV path uses precomputed features; live `/score` enriches per request.
- Throughput depends on hardware and SHAP frequency on flagged rows.
- ULB benchmark is a **separate** credit-card experiment, not a UPI production claim.

---

## Final checklist (this session)

| Check | Result |
|---|---|
| pytest (pinned venv from clean clone) | **5 passed** in 45.78s |
| `npm run build` (main `dashboard/`) | OK — vite built in 39.92s |
| `npm run build` (clean-clone `dashboard/`) | OK — vite built in 13.45s |
| Clean-clone `GET /metrics` | **200** after ~35s warm-up |
| `git push` | earlier `bd2135c..05923d7`; final docs/screenshot commit pushed after this report |
