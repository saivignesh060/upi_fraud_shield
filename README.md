# 🛡️ UPI FraudShield

An end-to-end, AI-powered real-time fraud detection and threat intelligence platform designed for Unified Payments Interface (UPI) transactions. 

UPI FraudShield combines advanced machine learning (XGBoost), Explainable AI (SHAP), and high-performance WebSockets to analyze streams of transactions millisecond-by-millisecond. It features a stunning, dark-mode React dashboard that provides Security Operations Center (SOC) teams with deep operational visibility, actionable alerts, and interactive spatial network visualizations.

---

## 🌟 Key Features

### 🧠 AI & Machine Learning Pipeline
- **Synthesizer Engine:** Custom data generation scripts to simulate realistic Indian UPI network traffic, including diverse behavioral patterns and sophisticated fraud vectors (`location_jump`, `velocity_spikes`, etc.).
- **XGBoost Classifier:** High-accuracy Supervised Machine Learning model trained on extracted time-based and graph-based heuristics.
- **Explainable AI (XAI):** Real-time model interpretability using **SHAP (SHapley Additive exPlanations)**. Every flagged transaction is accompanied by a Natural Language explanation of *why* the AI made its decision, highlighting top risk factors.

### ⚡ Real-Time Backend
- **FastAPI & WebSockets:** A highly scalable asynchronous Python backend. Uses WebSockets to broadcast processed transactions to connected clients in real-time.
- **REST Telemetry:** Exposes comprehensive API endpoints for system metrics computing true/false positive rates, precision, recall, and F1-score dynamically.

### 🌐 SOC Dashboard & Threat Intelligence
- **Live Network Feed:** Auto-scrolling, real-time feed of network traffic with instantaneous risk scoring.
- **Actionable Alerts Queue:** Prioritized queue of flagged transactions displaying SHAP breakdown bars and mitigation recommendations.
- **Geo-Spatial Telemetry:** Interactive India Map (`Leaflet.js` + OpenStreetMap tiles with a CSS dark filter) visualizing geographically anomalous transactions. Features animated quadratic Bezier arcs tracing `location_jump` attacks between cities.

---

## 🏗️ Architecture Stack

**Backend Engine:**
*   **Language:** Python 3.10+
*   **Framework:** FastAPI, Uvicorn
*   **ML Stack:** Scikit-Learn, XGBoost, Pandas, Numpy
*   **Explainability:** SHAP

**Frontend Dashboard:**
*   **Core:** React 18, TypeScript, Vite
*   **Styling:** TailwindCSS, Lucide React Icons
*   **Charts & Viz:** Recharts, Leaflet, React-Leaflet

---

## 📂 Project Structure

```text
upi_fraud_detector/
├── api/                     # FastAPI backend application & WebSocket router
├── dashboard/               # React (Vite) Frontend application
│   ├── src/                 # React components (GeoMap, ThreatIntelligence, etc.)
│   └── index.css            # Tailwind configuration & global styles
├── explainability/          # SHAP integration and Natural Language (NL) parsers
├── models/                  # Trained artifacts (XGBoost), encoders, and training scripts
├── generate_dataset.py      # Core UPI transaction synthesis tool
├── train_models.py          # ML model training pipeline
├── requirements.txt         # Python dependencies
├── upi_transactions.csv     # Generated locally (gitignored); required at project root
└── README.md                # Project documentation
```

---

## 🚀 Getting Started

Follow these steps to run the complete UPI FraudShield stack on your local machine.

### 1. Prerequisites
Ensure you have the following installed:
- Python 3.10+
- Node.js 18+ and npm

### 2. Backend Setup (AI & API)

Open a terminal and navigate to the project root:

```bash
# 1. Create and activate a Virtual Environment
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Windows CMD:
# .venv\Scripts\activate.bat
# Linux / macOS / Git Bash:
# source .venv/bin/activate   # or: source .venv/Scripts/activate on Windows Git Bash

# 2. Install Python Dependencies
pip install -r requirements.txt

# 3. Generate Dataset (Required) & Train Model (Optional)
# The pre-trained models are already included in the repository!
# However, the 100MB+ transaction CSV is excluded due to file size limits.
# You must generate the data locally before starting the server.
# Output path is the project root: upi_transactions.csv
python generate_dataset.py

# (Optional) If you modify the generated data or feature engineering,
# you can retrain the models from scratch:
# python train_models.py

# 4. Start the FastAPI Server
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
*The backend API will run on `http://localhost:8000`.*

### 3. Frontend Setup (React Dashboard)

Open a **new** terminal window and navigate to the `dashboard` directory:

```bash
cd dashboard

# 1. Install Node Dependencies
# package.json already pins react-leaflet@^4.2.1 for React 18 compatibility
npm install

# 2. Start the Vite Development Server
npm run dev
```
*The frontend dashboard will run on `http://localhost:5173`. Open this URL in your browser.*

---

## 📊 Measured model metrics (test set)

Leak-free training: **per-user chronological 70/15/15 split**, baselines from **train legitimate rows only**, threshold chosen on validation (max recall s.t. precision ≥ 0.5). Dataset uses `FRAUD_RATE=0.008` with hard negatives and overlapping fraud signals (see Limitations). Figures from `models/xgb_metrics.json` on this machine:

| Metric | Value |
|---|---|
| Decision threshold | **0.794784** |
| ROC-AUC | **0.8901** |
| PR-AUC | **0.5572** |
| Precision | **0.4993** |
| Recall | **0.5434** |
| F1 | **0.5204** |

Stream throughput after precomputed features + batch predict + monotonic scheduler (**500** messages): **14.20 msg/s** at `tps=20`, **48.49 msg/s** at `tps=50`.

### Limitations
- Data is **synthetic**, not production UPI traffic.
- Harder generator (~0.8% fraud, hard negatives, probabilistic/overlapping signatures, 1% train-only label noise) is still less messy than real fraud.
- Users/merchants are shared across time splits; scores can still look optimistic vs fully unseen entities.

---

## 🧭 Navigating the Platform

Once both servers are running, access the dashboard.
1. **Main Dashboard:** 
   - Monitor the **Real-time Volume** chart for overall traffic trends.
   - Watch the **Model Evaluation** panel to see live F1-Score and False Positive Rates.
   - Review the **Actionable Alerts** queue for SHAP explanations of intercepted attacks.
2. **Threat Intelligence Tab:** 
   - Click the target icon in the sidebar to switch views.
   - **GeoMap:** Watch live drops of transactions across India. Look for red pulsing arcs indicating impossible travel (Location Jumps).

---

## 🚢 Production Deployment

This section covers how to deploy UPI FraudShield beyond your local machine. The project has three distinct layers to deploy: the **Backend API**, the **Frontend Dashboard**, and the **ML Models**.

### 🔷 Backend API (FastAPI) — Deploy to Render / Railway / EC2

The FastAPI backend with the WebSocket stream can be deployed to any platform that supports Python.

**Recommended: [Render](https://render.com/) (free tier available)**

1. Push your code to GitHub (already done ✅).
2. Create a new **Web Service** on Render, connecting it to your `upi_fraud_shield` repository.
3. Set the following configuration:
   - **Runtime:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
4. The project root already includes `requirements.txt` (FastAPI, Uvicorn, pandas, numpy, scikit-learn, xgboost, shap, numba, websockets, python-multipart, joblib).
5. Before deploying, generate the dataset at the **project root** (`python generate_dataset.py` → `upi_transactions.csv`) and ensure the `models/` folder is pushed to the repo (already done ✅).

> **Note on Models:** The pre-trained XGBoost model files are already included in the repository. The backend will load them directly from disk on startup — **no retraining needed on the server**.

> **Important on Data:** `upi_transactions.csv` is not shipped in the repo. The backend streams from this CSV at runtime. On a production server, either run `generate_dataset.py` once post-deploy via a shell command, or modify the API to use a database instead.

---

### 🔷 Frontend Dashboard (React) — Deploy to Vercel

The Vite + React frontend can be deployed for free on [Vercel](https://vercel.com/).

1. Go to [vercel.com](https://vercel.com/) and import your GitHub repository.
2. Set the **Root Directory** to `dashboard`.
3. Vercel will auto-detect Vite and configure the build. Confirm:
   - **Build Command:** `npm run build`
   - **Output Directory:** `dist`
4. Add an **Environment Variable** pointing to your deployed backend URL:
   ```
   VITE_API_URL=https://your-render-backend-url.onrender.com
   ```
5. `dashboard/src/App.tsx` already reads `VITE_API_URL` and derives the WebSocket URL (`https` → `wss`, then `http` → `ws`), defaulting to `http://localhost:8000` for local dev.
6. Deploy — Vercel handles the rest and gives you a public `*.vercel.app` URL.

---

### 🔷 ML Models — No Separate Deployment Needed

The XGBoost classifier and label encoders are `.json` / `.pkl` files bundled directly inside the repository and loaded by the FastAPI backend at startup. There is **no separate model server** required. The backend IS the model inference server.

If you scale to high traffic in the future, consider:
- Wrapping the model in a dedicated microservice
- Using **ONNX** for cross-platform compatibility
- Caching predictions by transaction signature using **Redis**

---

## 🛡️ License & Terms
Developed for demonstration and operational prototyping.
