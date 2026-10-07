# ⚡ Olist E-Commerce Decision Intelligence & Event Dispatch Gateway

> **An end-to-end dual-pipeline machine learning system for turning Olist marketplace signals into operational decisions, customer-retention actions, and real-time event alerts.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](#)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)](#)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](#)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.3%2B-brightgreen?style=for-the-badge)](#)
[![Discord](https://img.shields.io/badge/Discord-Async_Webhooks-5865F2?style=for-the-badge&logo=discord&logoColor=white)](#)
[![License](https://img.shields.io/badge/License-MIT-black?style=for-the-badge)](#)

---
# Live demo

URL = https://olist-customer-intelligence-dashboard.streamlit.app/

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Business Problem](#-business-problem)
- [What the System Does](#-what-the-system-does)
- [Architecture](#-architecture)
- [Machine Learning Pipelines](#-machine-learning-pipelines)
- [Operational Decision Engine](#-operational-decision-engine)
- [Repository Structure](#-repository-structure)
- [Tech Stack](#-tech-stack)
- [Local Setup](#-local-setup)
- [Running the Services](#-running-the-services)
- [API Reference](#-api-reference)
- [Discord Alerting](#-discord-alerting)
- [Cloud Deployment](#-cloud-deployment)
- [Production Considerations](#-production-considerations)
- [License](#-license)

---

## 🔎 Overview

This project combines **machine learning, API engineering, dashboarding, and event-driven automation** into a single decision-intelligence platform built around the Olist Brazilian marketplace dataset.

The system contains two independent prediction pipelines:

1. **Pre-Dispatch Friction Risk**
   - Predicts whether an order is likely to result in a dissatisfied review.
   - Converts the prediction into logistics-oriented intervention decisions.

2. **VIP Repeat Buyer Propensity**
   - Identifies first-order customers with a higher likelihood of placing another order.
   - Converts the propensity score into customer-retention and CRM actions.

Predictions are exposed through a **FastAPI inference gateway**, visualized through a **Streamlit decision dashboard**, and routed to operational channels through **Discord webhooks**.

---

## 🎯 Business Problem

Marketplace operations face two complementary problems:

### 1. Post-purchase dissatisfaction

According to the project's analysis, orders with review scores **≤ 2** represent a meaningful dissatisfaction risk. Poor delivery economics, long transit times, package dimensions, weight, and freight friction can contribute to operational intervention needs.

### 2. Customer retention

The project analysis reports a **3.1% natural repeat-purchase baseline** and focuses on identifying customers whose first-order characteristics indicate stronger repeat-purchase potential.

### The objective

Instead of stopping at model predictions, this system converts predictions into **operational actions**:

| Business Signal | ML Output | Operational Response |
|---|---|---|
| High delivery-friction risk | Dissatisfaction probability | Logistics intervention |
| Bulky / heavy shipment | Risk + feature diagnosis | Repackaging audit |
| Long transit + high freight ratio | High-risk diagnosis | Priority / express routing |
| High repeat propensity | Repeat probability | CRM personalization |
| High first-order spend | VIP diagnosis | Executive-tier treatment |
| Multi-item first order | VIP diagnosis | Cross-category promotion |

---

## 🚀 What the System Does

### Pre-Dispatch Risk Pipeline

Predicts dissatisfaction risk **before carrier handoff** and determines whether an order should remain in standard fulfillment or receive operational intervention.

### VIP Repurchase Pipeline

Scores first-order customers for repeat-purchase potential and identifies high-potential prospects for targeted retention workflows.

### Decision Layer

Model probabilities are not exposed as raw numbers alone. They are translated into business-friendly decisions such as:

- `Standard Fulfillment`
- `High Friction Risk`
- `Priority Logistics`
- `Warehouse Repackaging Audit`
- `High-LTV VIP Prospect`
- `Executive Tier`

### Event Dispatch Layer

High-priority events can be sent asynchronously to Discord channels:

- `#warehouse-alerts`
- `#crm-vip-leads`

---

# 🏗 Architecture

```text
                         ┌─────────────────────────────────┐
                         │   Streamlit Executive Dashboard │
                         │  Plotly Gauges + Diagnosis UI   │
                         └───────────────┬─────────────────┘
                                         │
                                  HTTP POST / JSON
                                         │
                                         ▼
                         ┌─────────────────────────────────┐
                         │     FastAPI Prediction Gateway  │
                         │      Pydantic V2 Validation     │
                         └───────────────┬─────────────────┘
                                         │
                         ┌───────────────┴───────────────┐
                         │                               │
                         ▼                               ▼
              ┌──────────────────────┐       ┌──────────────────────┐
              │ Satisfaction Pipeline │       │ Repeat Buyer Pipeline│
              │      LightGBM        │       │       LightGBM       │
              │ τ = 0.5397           │       │ τ = 0.0500           │
              └──────────┬───────────┘       └──────────┬───────────┘
                         │                               │
                         └───────────────┬───────────────┘
                                         │
                                  Decision / Action
                                         │
                                         ▼
                         ┌─────────────────────────────────┐
                         │     Discord Event Router        │
                         │     Non-blocking Webhooks       │
                         └───────────────┬─────────────────┘
                                         │
                         ┌───────────────┴───────────────┐
                         ▼                               ▼
                #warehouse-alerts                #crm-vip-leads
```

### Architecture Flow

```text
User / Dashboard
      │
      ▼
Streamlit
      │
      ▼
FastAPI
      │
      ├──────────────► Satisfaction Model
      │                       │
      │                       ▼
      │                 Risk Diagnosis
      │
      └──────────────► Repeat Purchase Model
                              │
                              ▼
                       VIP Diagnosis
                              │
                              ▼
                       Action Router
                              │
                              ▼
                       Discord Webhooks
```

---

# 🤖 Machine Learning Pipelines

## Pipeline 1 — Pre-Dispatch Friction Risk

**Model artifact:** `lgbm_dissatisfaction_pipeline.pkl`

### Objective

Binary classification for predicting whether an order is likely to receive a review score of **≤ 2 stars** before transit dispatch.

### Decision Threshold

```text
τ = 0.5397
```

The project uses this calibrated threshold as its operational decision boundary.

### Pipeline

```text
Raw Features
     │
     ▼
ColumnTransformer
     ├── Numerical preprocessing
     └── Categorical One-Hot Encoding
     │
     ▼
LightGBM Classifier
     │
     ▼
Dissatisfaction Probability
     │
     ▼
Operational Threshold
     │
     ▼
Action / Alert
```

### Feature Importance

| Rank | Feature | Split Importance | Operational Interpretation |
|---:|---|---:|---|
| 1 | `total_volume_cm3` | 906 | Package cubic dimensions |
| 2 | `estimated_transit_days` | 782 | Expected carrier delivery window |
| 3 | `total_order_price` | 740 | Total buyer order value |
| 4 | `total_weight_g` | 708 | Shipment weight |
| 5 | `total_freight_value` | 655 | Freight charged to the order |
| 6 | `freight_to_price_ratio` | 592 | Freight cost relative to order value |
| 7 | `purchase_month` | 440 | Seasonal operational effects |
| 8 | `purchase_hour` | 402 | Order timing / dispatch cutoff effects |
| 9 | `payment_installments_max` | 220 | Buyer installment exposure |
| 10 | `purchase_dayofweek` | 186 | Weekly processing effects |

---

## Pipeline 2 — VIP Repeat Buyer Engine

**Model artifact:** `lgbm_repeat_purchase_pipeline.pkl`

### Objective

Binary classification for identifying customers likely to place **≥ 2 lifetime orders**.

### Baseline

```text
Natural repeat-purchase baseline = 3.1%
```

### Decision Threshold

```text
τ = 0.0500
```

The project uses this threshold to identify high-potential repeat buyers.

### Feature Schema

```text
first_order_spend
first_order_freight
first_order_freight_ratio
first_order_items_count
first_order_installments
first_order_payment_type
customer_state
```

### Pipeline

```text
First-Order Customer Data
          │
          ▼
Feature Transformation
          │
          ▼
LightGBM Classifier
          │
          ▼
Repeat-Purchase Probability
          │
          ▼
VIP Threshold
          │
          ▼
CRM Action
```

---

# 🧠 Operational Decision Engine

The system combines model probabilities with business rules to produce an actionable diagnosis.

## A. Pre-Dispatch Friction Remediation

### Low Risk

```text
Risk < 0.5397
```

**Status:** Optimal Delivery Clearance

**Action:** Clear for standard automated fulfillment.

### High Risk

```text
Risk ≥ 0.5397
```

The system then applies additional feature-based diagnosis.

#### Case 1 — High Freight Ratio + Extended SLA

```text
freight_ratio > 0.40
AND
transit >= 25 days
```

**Action:** Reroute to a Tier-1 express carrier and trigger proactive tracking.

#### Case 2 — Bulky Package

```text
volume >= 15000 cm³
OR
weight >= 3000 g
```

**Action:** Trigger a physical warehouse repackaging audit.

#### Case 3 — Standard Friction

**Action:** Flag the order for priority fulfillment and verify carrier dispatch within 4 hours.

---

## B. VIP Customer Personalization

### Standard Buyer

```text
Propensity < 0.0500
```

**Action:** Keep the customer in the standard retention / newsletter workflow.

### High-Potential Buyer

```text
Propensity ≥ 0.0500
```

#### Case 1 — High Spend

```text
first_order_spend >= R$ 250
```

**Action:** Enroll into Executive Tier and dispatch a R$30 thank-you voucher within 72 hours.

#### Case 2 — Multi-Item First Order

```text
first_order_items_count > 1
```

**Action:** Trigger a cross-category bundle promotion for the next checkout cycle.

#### Case 3 — Standard VIP

**Action:** Dispatch a targeted 10% repeat-purchase voucher and apply priority customer-support tagging.

---

# 📂 Repository Structure

```text
olist-customer-intelligence/
│
├── api/
│   ├── __init__.py
│   ├── main.py
│   └── schemas.py
│
├── dashboard/
│   ├── __init__.py
│   └── app.py
│
├── models/
│   ├── lgbm_dissatisfaction_pipeline.pkl
│   └── lgbm_repeat_purchase_pipeline.pkl
│
├── data/
│   └── processed/
│
├── notebooks/
│   └── # Feature engineering + model development
│
├── .env.example
├── .gitignore
├── requirements.txt
├── LICENSE
└── README.md
```

### Core Components

| Component | Responsibility |
|---|---|
| `api/main.py` | FastAPI inference service + event routing |
| `api/schemas.py` | Pydantic V2 request / response contracts |
| `dashboard/app.py` | Streamlit decision dashboard |
| `models/` | Serialized ML pipelines |
| `data/processed/` | Processed analytical data |
| `notebooks/` | Feature engineering and training workflows |
| `.env.example` | Environment-variable template |
| `requirements.txt` | Python dependencies |

---

# 🧰 Tech Stack

### Machine Learning

- Python
- Pandas
- NumPy
- Scikit-learn
- LightGBM

### API & Backend

- FastAPI
- Pydantic V2
- Uvicorn

### Dashboard

- Streamlit
- Plotly

### Event Integration

- Discord Webhooks
- Asynchronous HTTP requests

### Deployment

- Render
- Streamlit Community Cloud

---

# ⚙️ Local Setup

## 1. Clone the Repository

```bash
git clone https://github.com/<durgesh693 >/olist-customer-intelligence.git
cd olist-customer-intelligence
```

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
# Discord Webhooks
DISCORD_WAREHOUSE_WEBHOOK_URL="https://discord.com/api/webhooks/your_warehouse_webhook_url"
DISCORD_CRM_WEBHOOK_URL="https://discord.com/api/webhooks/your_crm_webhook_url"

# Local API endpoints
SATISFACTION_API_URL="http://localhost:8000/predict/satisfaction"
REPEAT_PURCHASE_API_URL="http://localhost:8000/predict/repeat-purchase"
```

> **Security:** Never commit real webhook URLs, API keys, passwords, or other secrets to GitHub.

---

# ▶️ Running the Services

The application uses two local services.

## Terminal 1 — FastAPI

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### API Documentation

```text
http://localhost:8000/docs
```

### Health Check

```text
http://localhost:8000/
```

---

## Terminal 2 — Streamlit

```bash
streamlit run dashboard/app.py
```

### Dashboard

```text
http://localhost:8501
```

---

# 📡 API Reference

## 1. Satisfaction Prediction

### Endpoint

```http
POST /predict/satisfaction
```

### Request

```json
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

### Response

```json
{
  "dissatisfaction_risk": 0.6128,
  "alert_triggered": true,
  "recommended_action": "Priority logistics reroute & mandatory warehouse repackaging audit.",
  "risk_tier": "High Friction Risk"
}
```

---

## 2. VIP Repeat Purchase

### Endpoint

```http
POST /predict/repeat-purchase
```

### Request

```json
{
  "first_order_spend": 180.0,
  "first_order_freight": 25.0,
  "first_order_items_count": 1,
  "first_order_installments": 2,
  "first_order_payment_type": "credit_card",
  "customer_state": "SP"
}
```

### Response

```json
{
  "repeat_probability": 0.0824,
  "is_high_potential": true,
  "crm_action": "Enroll into Executive Tier. Dispatch an unprompted R$ 30 thank-you voucher within 72 hours.",
  "customer_tier": "High-LTV VIP Prospect"
}
```

---

# 🔔 Discord Alerting

The platform can route operational events to separate Discord webhook channels.

### Warehouse Channel

```text
#warehouse-alerts
```

### CRM Channel

```text
#crm-vip-leads
```

## Verify Warehouse Webhook

```bash
curl \
  -H "Content-Type: application/json" \
  -X POST \
  -d '{"content":"🚀 Warehouse Channel Verified"}' \
  YOUR_WAREHOUSE_WEBHOOK_URL
```

## Verify CRM Webhook

```bash
curl \
  -H "Content-Type: application/json" \
  -X POST \
  -d '{"content":"🌟 CRM Channel Verified"}' \
  YOUR_CRM_WEBHOOK_URL
```

---

# ☁️ Cloud Deployment

## Part A — Deploy FastAPI to Render

1. Open the Render Dashboard.
2. Select **New → Web Service**.
3. Connect the GitHub repository.
4. Configure:

```text
Runtime:
Python 3

Build Command:
pip install -r requirements.txt

Start Command:
uvicorn api.main:app --host 0.0.0.0 --port $PORT
```

5. Add production environment variables:

```text
DISCORD_WAREHOUSE_WEBHOOK_URL=<your_warehouse_url>
DISCORD_CRM_WEBHOOK_URL=<your_crm_url>
```

6. Deploy the service.
7. Copy the generated API URL.

Example:

```text
https://olist-intelligence-api.onrender.com
```

---

## Part B — Deploy Streamlit

1. Open Streamlit Community Cloud.
2. Create a new application.
3. Select the repository and `main` branch.
4. Set the main file:

```text
dashboard/app.py
```

5. Add production secrets:

```toml
SATISFACTION_API_URL = "https://olist-intelligence-api.onrender.com/predict/satisfaction"
REPEAT_PURCHASE_API_URL = "https://olist-intelligence-api.onrender.com/predict/repeat-purchase"
```

6. Deploy the application.

---

# 🛡 Production Considerations

Before treating the project as a production service, consider adding:

- Authentication and authorization for prediction endpoints.
- Request validation and rate limiting.
- Structured application logging.
- Model versioning and artifact tracking.
- Monitoring for prediction drift and data drift.
- Centralized error handling.
- Webhook retry and failure handling.
- Secret management through the deployment platform.
- CI/CD checks for tests, linting, and deployment.
- Automated model-performance monitoring.

These items are recommended engineering improvements rather than claims about the current implementation.

---

# 📊 Project Highlights

| Capability | Implementation |
|---|---|
| Dual ML pipelines | LightGBM |
| Prediction API | FastAPI |
| Contract validation | Pydantic V2 |
| Interactive analytics | Streamlit + Plotly |
| Operational diagnosis | Threshold + rule engine |
| Event dispatch | Discord Webhooks |
| Local deployment | Uvicorn |
| Cloud API deployment | Render |
| Dashboard deployment | Streamlit Community Cloud |
| License | MIT |

---


## ⭐ Project Philosophy

> **Predict → Diagnose → Decide → Dispatch**

The goal of this project is not simply to build machine-learning models. It is to demonstrate how model outputs can be connected to **real operational decisions, APIs, dashboards, and event-driven workflows**.

---

## 👤 Author

**Durgesh maurya**
