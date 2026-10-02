# ⚡ Olist E-Commerce Decision Intelligence & Event Dispatch Gateway

> **An End-to-End, Production-Grade Dual-Pipeline Machine Learning System engineered on 100,000+ Brazilian Marketplace Orders.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](#)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)](#)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](#)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.3%2B-brightgreen?style=for-the-badge)](#)
[![Discord](https://img.shields.io/badge/Discord-Async_Webhooks-5865F2?style=for-the-badge&logo=discord&logoColor=white)](#)

---

## 📌 1. Executive Summary & Problem Statement

In large-scale marketplace platforms like **Olist (Brazil)**, post-purchase dissatisfaction creates massive downstream operational costs:
* **The Post-Dispatch Support Trap:** Once an order leaves the fulfillment center with poor economics or extended transit times, customer dissatisfaction (review scores $\le 2$) spikes to **14.8%**, driving support escalations, return shipping overhead, and negative seller ratings.
* **The Retention Deficit:** Marketplace repurchasing sits at an organic baseline of only **3.1%**. Generic batch promotions fail because retention propensity is concentrated in specific first-order basket traits.

### System Solution
This platform bridges analytical modeling directly with live fulfillment and growth operations:
1. **Pre-Dispatch Friction Mitigation:** Flags high-risk orders **before carrier handoff**, triggering automated repackaging audits or priority courier upgrades.
2. **VIP Repurchase Propensity Engine:** Identifies high-value repeat prospects on their very first purchase for automated loyalty workflows.
3. **Decoupled Asynchronous Alerting:** Pushes real-time alerts into operational communication channels via Discord Webhook gateways (`#warehouse-alerts` and `#crm-vip-leads`).

---

## 🏛 2. System Architecture & Data Flow

```text
                                  +---------------------------------------+
                                  |   Streamlit Executive Dashboard UI    |
                                  |   (Plotly Gauges & Dynamic Diagnosis) |
                                  +-------------------+-------------------+
                                                      |
                                          HTTP POST   |   JSON Payloads
                                          (Port 8501) |
                                                      v
                                  +---------------------------------------+
                                  |      FastAPI Prediction Gateway       |
                                  |   (Pydantic V2 Contract Validation)   |
                                  +---------+-------------------+---------+
                                            |                   |
                        Inference Request   |                   | Inference Request
                                            v                   v
                    +---------------------------+   +---------------------------+
                    | Satisfaction Pipeline     |   | Repeat Purchase Pipeline  |
                    | LightGBM Pre-Dispatch     |   | LightGBM VIP Classifier   |
                    | Threshold: tau >= 0.5397  |   | Threshold: tau >= 0.0500  |
                    +-------------+-------------+   +-------------+-------------+
                                  |                               |
                     Risk >= 0.5397 (Alert Trigger)  Propensity >= 0.05 (VIP Match)
                                  +---------------+---------------+
                                                  |
                                                  v  (Non-blocking Async POST)
                                  +---------------------------------------+
                                  |       Discord Event Router Layer      |
                                  +---------+-------------------+---------+
                                            |                   |
                                            v                   v
                                 [ #warehouse-alerts ]  [ #crm-vip-leads ]


## 🔬 3. Machine Learning Pipelines & Empirical Schema

### Pipeline 1: Pre-Dispatch Friction Risk (`lgbm_dissatisfaction_pipeline.pkl`)

- **Target Objective:** Binary classification predicting Review Score $\le 2$ Stars prior to transit dispatch.
- **Calibrated Decision Boundary:** $\tau = 0.5397$ (Cost-optimized to minimize operational intervention costs against churn loss).
- **Pipeline Structure:** Scikit-Learn `ColumnTransformer` (Numerical standard scaling, categorical one-hot encoding) + `LGBMClassifier`.

#### Empirical Feature Importance Ranking (Model Ground Truth):

| **Rank** | **Feature Identifier**     | **Split Importance** | **Operational Interpretation**                                                             |
| -------- | -------------------------- | -------------------- | ------------------------------------------------------------------------------------------ |
| **1**    | `total_volume_cm3`         | **906**              | Package cubic dimensions; oversized items cause sorting and handling delays.               |
| **2**    | `estimated_transit_days`   | **782**              | Carrier SLA delivery window promised to customer.                                          |
| **3**    | `total_order_price`        | **740**              | Total monetary value committed by the buyer.                                               |
| **4**    | `total_weight_g`           | **708**              | Physical weight overhead impacting multi-tier linehaul handling.                           |
| **5**    | `total_freight_value`      | **655**              | Delivery charge billed to the order.                                                       |
| **6**    | `freight_to_price_ratio`   | **592**              | Cost friction metric computed dynamically as: $\frac{\text{Freight}}{\text{Order Price}}$. |
| **7**    | `purchase_month`           | **440**              | Seasonal carrier hub congestion and holiday peaks.                                         |
| **8**    | `purchase_hour`            | **402**              | Diurnal order submission window and same-day dispatch cutoffs.                             |
| **9**    | `payment_installments_max` | **220**              | Buyer credit sensitivity and split payment exposure.                                       |
| **10**   | `purchase_dayofweek`       | **186**              | Weekend processing backlogs at intake docks.                                               |

### Pipeline 2: VIP Repeat Buyer Engine (`lgbm_repeat_purchase_pipeline.pkl`)

- **Target Objective:** Binary classification identifying buyers likely to place $\ge 2$ lifetime orders.
- **Natural Baseline:** $3.1%$ natural marketplace repeat rate.
- **VIP Target Boundary:** $\tau \ge 0.0500$ (Captures top decile prospects with a 3.4x precision lift over random outreach).
- **Feature Schema:** `first_order_spend`, `first_order_freight`, `first_order_freight_ratio`, `first_order_items_count`, `first_order_installments`, `first_order_payment_type`, `customer_state`.

## ⚙️ 4. Dynamic Diagnosis & Operational Playbooks

The system translates model probabilities into actionable playbooks based on input features:

### A. Pre-Dispatch Friction Remediation (Tab 4):

- **Friction Score $< 0.5397$:**
  - **Status:** Optimal Delivery Clearance.
  - **Playbook:** Clear for automated standard fulfillment line.
- **Friction Score $\ge 0.5397$:**
  - **Case 1 (High Ratio & Extended SLA):** If `freight_ratio > 0.40` AND `transit >= 25 days`:
    - *Action:* Reroute package to Tier-1 Express Carrier and trigger proactive SMS tracking.
  - **Case 2 (Bulky Package):** If `volume >= 15000 cm³` OR `weight >= 3000 g`:
    - *Action:* Execute physical warehouse repackaging audit to reinforce protective cushioning.
  - **Case 3 (Standard Friction):**
    - *Action:* Flag priority fulfillment queue; verify carrier dispatch within 4 hours.

### B. VIP Customer Personalization (Tab 3):

- **Propensity Score $< 0.0500$:**
  - **Status:** Standard Single-Order Buyer.
  - **Playbook:** Retain in standard automated quarterly newsletter loop.
- **Propensity Score $\ge 0.0500$:**
  - **Case 1 (High Spend):** If `first_order_spend >= R$ 250`:
    - *Action:* Enroll into Executive Tier. Dispatch an unprompted R$ 30 thank-you voucher within 72 hours.
  - **Case 2 (Multi-Item Order):** If `first_order_items_count > 1`:
    - *Action:* Trigger cross-category bundle discount campaign for next checkout cycle.
  - **Case 3 (Standard VIP):**
    - *Action:* Dispatch targeted 10% repeat purchase voucher + Priority customer support tagging.

## 📂 5. Repository Structure

```
olist-customer-intelligence/
├── api/
│   ├── __init__.py
│   ├── main.py                  # FastAPI service with non-blocking Discord routing
│   └── schemas.py               # Pydantic V2 request & response schemas
├── dashboard/
│   ├── __init__.py
│   └── app.py                   # Streamlit decision hub & Plotly gauge UI
├── models/
│   ├── lgbm_dissatisfaction_pipeline.pkl  # Pre-dispatch friction model
│   └── lgbm_repeat_purchase_pipeline.pkl  # First-order VIP retention model
├── data/
│   └── processed/               # Cleaned analytical cohorts
├── notebooks/                   # Feature engineering & training pipelines
├── .env.example                 # Environment configuration template
├── .gitignore                   # Deployment exclusions & secret guards
├── requirements.txt             # Exact locked dependencies
└── README.md                    # Platform documentation

```

## 🛠 6. Local Setup & Installation

### Step 1: Clone Repository

```
git clone https://github.com/<your-username>/olist-customer-intelligence.git
cd olist-customer-intelligence

```

### Step 2: Create & Activate Virtual Environment

```
# Windows (Command Prompt / Git Bash)
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate

```

### Step 3: Install Dependencies

```
pip install -r requirements.txt

```

### Step 4: Configure Environment Variables

Create a `.env` file in the project root:

```
# Discord Webhook Gateways
DISCORD_WAREHOUSE_WEBHOOK_URL="https://discord.com/api/webhooks/your_warehouse_webhook_url"
DISCORD_CRM_WEBHOOK_URL="https://discord.com/api/webhooks/your_crm_webhook_url"

# API Endpoints (Localhost defaults)
SATISFACTION_API_URL="http://localhost:8000/predict/satisfaction"
REPEAT_PURCHASE_API_URL="http://localhost:8000/predict/repeat-purchase"

```

## 🚀 7. Running the Microservices Locally

Launch both services in separate terminal windows:

### Terminal 1: FastAPI Inference Core

```
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

```

- **Interactive Swagger UI:** `http://localhost:8000/docs`
- **Health Check Probe:** `http://localhost:8000/`

### Terminal 2: Streamlit Analytics Hub

```
streamlit run dashboard/app.py

```

- **Dashboard Access:** `http://localhost:8501`

## 📡 8. API Contract Specifications

### 1. Satisfaction Prediction Endpoint

- **Method & Route:** `POST /predict/satisfaction`
- **Request Payload:**

```
{
  "total_order_price": 85.0,
  "total_freight_value": 45.0,
  "total_volume_cm3": 12000.0,
  "total_weight_g": 2500.0,
  "estimated_transit_days": 28.0,
  "is_cross_state": 1,
  "payment_installments_max": 2
}

```

- **Response Payload (`200 OK`):**

```
{
  "dissatisfaction_risk": 0.6128,
  "alert_triggered": true,
  "recommended_action": "Priority logistics reroute & mandatory warehouse repackaging audit.",
  "risk_tier": "High Friction Risk"
}

```

### 2. VIP Repeat Buyer Endpoint

- **Method & Route:** `POST /predict/repeat-purchase`
- **Request Payload:**

```
{
  "first_order_spend": 180.0,
  "first_order_freight": 25.0,
  "first_order_items_count": 1,
  "first_order_installments": 2,
  "first_order_payment_type": "credit_card",
  "customer_state": "SP"
}

```

- **Response Payload (`200 OK`):**

```
{
  "repeat_probability": 0.0824,
  "is_high_potential": true,
  "crm_action": "Enroll into Executive Tier. Dispatch an unprompted R$ 30 thank-you voucher within 72 hours.",
  "customer_tier": "High-LTV VIP Prospect"
}

```

## 🔔 9. Discord Alerting Integration & Verification

To verify that the Discord webhook endpoints are receiving requests, test via terminal:

```
# Test Warehouse Alert Channel
curl -H "Content-Type: application/json" -X POST -d '{"content": "🚀 Warehouse Channel Verified"}' YOUR_WAREHOUSE_WEBHOOK_URL

# Test CRM VIP Channel
curl -H "Content-Type: application/json" -X POST -d '{"content": "🌟 CRM Channel Verified"}' YOUR_CRM_WEBHOOK_URL

```

## ☁️ 10. Production Cloud Deployment Blueprint

### Part A: Deploy FastAPI to Render

1. Go to [**Render Dashboard**](https://dashboard.render.com/) and click **New + > Web Service**.
2. Connect your GitHub repository.
3. Set the configuration:
   - **Name:** `olist-intelligence-api`
   - **Runtime:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
4. Add **Environment Variables**:
   - `DISCORD_WAREHOUSE_WEBHOOK_URL` = `<your_warehouse_url>`
   - `DISCORD_CRM_WEBHOOK_URL` = `<your_crm_url>`
5. Click **Deploy Web Service** and copy your live URL (e.g., `https://olist-intelligence-api.onrender.com`).

### Part B: Deploy Streamlit to Streamlit Community Cloud

1. Go to [**share.streamlit.io**](https://share.streamlit.io/) and click **New app**.
2. Select your repository, branch (`main`), and set **Main file path** to:

   Plaintext
   ```
   dashboard/app.py

   ```
3. Open **Advanced settings > Secrets** and paste your production URLs:

   Ini, TOML
   ```
   SATISFACTION_API_URL = "https://olist-intelligence-api.onrender.com/predict/satisfaction"
   REPEAT_PURCHASE_API_URL = "https://olist-intelligence-api.onrender.com/predict/repeat-purchase"

   ```
4. Click **Deploy!**

## 📜 11. License

Distributed under the **MIT License**. See `LICENSE` for details.
