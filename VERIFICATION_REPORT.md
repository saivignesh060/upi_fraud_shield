# UPI FraudShield — Verification Report (final)

**Date:** 2026-11-06  
**Method:** Commands executed; only measured numbers stated.

---

## Profile: one scored transaction (before stream optimization)

| Stage | ms |
|---|---|
| enrich_profile | 0.061 |
| **feature_build (slowest)** | **172.569** |
| predict | 27.280 |
| shap (flagged) | 15.820 |
| json_serialize | 0.304 |
| total_compute | 216.035 |
| sleep budget @ tps=20 | 50.000 |

Fix applied: precompute CSV feature matrix once, batch `predict_proba` (256), monotonic-clock scheduler (`target_time += 1/tps`).

---

## Throughput (500 messages, measured)

| Requested `tps` | Before batch/precompute | After |
|---|---|---|
| 20 | 8.69 msg/s (n=200) | **14.20 msg/s** |
| 50 | 12.66 msg/s (n=200) | **48.49 msg/s** |

---

## Model metrics — leak-free split, before vs after harder data

| Metric | Before harder data (fraud 3.2%) | After harder data (fraud 0.80%) |
|---|---|---|
| Threshold (val: max recall s.t. P≥0.5) | 0.462876 | **0.794784** |
| ROC-AUC | 0.9935 | **0.8901** |
| PR-AUC | 0.9023 | **0.5572** |
| Precision | 0.4606 | **0.4993** |
| Recall | 0.9779→0.9800 | **0.5434** |
| F1 | 0.6267 | **0.5204** |
| Test N | 79,719 | 79,724 |

Hardening in `generate_dataset.py`: `FRAUD_RATE=0.008`, hard negatives (travel / night / large amt / new device labeled legit), ~70% signature strength on fraud patterns with overlapping distributions, **1% train-only label noise**.

---

## Feature checklist

| Item | Result |
|---|---|
| 9. NetworkGraph removed | PASS — GeoMap-only Threat Intelligence; `d3` / `@types/d3` removed; `npm run build` + `tsc` OK |
| 10. Throughput optimization | PASS — profile + batch/precompute/scheduler; rates above |
| 11. pytest + build + docs + commit | PASS (this report) |
| 12. Harder synthetic data + retrain | PASS — metrics table above |

Screenshot: `docs/geomap_ti_final.png` (Threat Intelligence, GeoMap only).

pytest: **5 passed**.

---

## Limitations

- Data is **synthetic** (Indian UPI-like generator), not production bank traffic.
- Harder mode still uses shared users/merchants across time splits and injected patterns; real fraud is noisier and adversarial.
- Train label noise (1%) is artificial; test labels remain clean for evaluation honesty.
- Stream scores use **precomputed** features for the CSV replay path (batch XGBoost); live `/score` still enriches from the profile engine per request.
- Throughput depends on hardware and how many alerts trigger SHAP.
