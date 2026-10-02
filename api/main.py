"""
FastAPI Service Gateway - Olist Customer Intelligence
====================================================
Exposes prediction endpoints and triggers silent Discord webhooks.
"""

import os
import joblib
import pandas as pd
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    SatisfactionInput, SatisfactionOutput,
    RepeatPurchaseInput, RepeatPurchaseOutput
)

# Load environment variables from .env
load_dotenv()

DISCORD_WAREHOUSE_WEBHOOK_URL = os.getenv("DISCORD_WAREHOUSE_WEBHOOK_URL")
DISCORD_CRM_WEBHOOK_URL = os.getenv("DISCORD_CRM_WEBHOOK_URL")

# Resolve Paths to Trained Models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

SATISFACTION_MODEL_PATH = os.path.join(MODELS_DIR, "lgbm_dissatisfaction_pipeline.pkl")
REPEAT_PURCHASE_MODEL_PATH = os.path.join(MODELS_DIR, "lgbm_repeat_purchase_pipeline.pkl")

# Load Models
try:
    satisfaction_pipeline = joblib.load(SATISFACTION_MODEL_PATH)
except Exception as e:
    satisfaction_pipeline = None
    print(f"[WARNING] Could not load satisfaction model: {e}")

try:
    repeat_pipeline = joblib.load(REPEAT_PURCHASE_MODEL_PATH)
except Exception as e:
    repeat_pipeline = None
    print(f"[WARNING] Could not load repeat purchase model: {e}")

# Decision Boundaries
FRICTION_THRESHOLD = 0.5397
VIP_THRESHOLD = 0.0500

app = FastAPI(
    title="Olist Intelligence API",
    description="Inference Gateway & Event Dispatch for Olist Marketplace Models",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def dispatch_discord_alert(webhook_url: str, title: str, description: str, color: int, fields: list):
    """Silently dispatches Discord webhook embed alert."""
    if not webhook_url or not webhook_url.startswith("http"):
        return
    
    payload = {
        "username": "Olist Intelligence Bot",
        "embeds": [{
            "title": title,
            "description": description,
            "color": color,
            "fields": fields,
            "footer": {"text": "Olist Automated Ops Gateway"}
        }]
    }
    try:
        requests.post(webhook_url, json=payload, timeout=3)
    except Exception as e:
        print(f"[ERROR] Failed to send Discord alert: {e}")


@app.get("/")
def health_check():
    return {
        "status": "online",
        "models_loaded": {
            "satisfaction": satisfaction_pipeline is not None,
            "repeat_purchase": repeat_pipeline is not None
        }
    }


# ==========================================
# 1. Satisfaction / Friction Risk Endpoint
# ==========================================
@app.post("/predict/satisfaction", response_model=SatisfactionOutput)
def predict_satisfaction(payload: SatisfactionInput):
    data = payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()
    
    # Auto-calculate ratio if missing or zero
    if not data.get("freight_to_price_ratio") or data["freight_to_price_ratio"] == 0:
        price = data["total_order_price"]
        freight = data["total_freight_value"]
        data["freight_to_price_ratio"] = freight / price if price > 0 else 0.0
        
    df = pd.DataFrame([data])
    
    if satisfaction_pipeline is not None:
        try:
            # Predict probabilities
            risk_prob = float(satisfaction_pipeline.predict_proba(df)[0][1])
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
    else:
        # Fallback simulation if model file missing
        risk_prob = 0.61 if data["freight_to_price_ratio"] > 0.4 or data["estimated_transit_days"] > 25 else 0.25

    alert_triggered = risk_prob >= FRICTION_THRESHOLD
    
    if alert_triggered:
        action = "Priority logistics reroute & mandatory warehouse repackaging audit."
        tier = "High Friction Risk"
        
        # Trigger silent warehouse webhook
        fields = [
            {"name": "Risk Probability", "value": f"{risk_prob*100:.1f}%", "inline": True},
            {"name": "Transit Window", "value": f"{data['estimated_transit_days']} Days", "inline": True},
            {"name": "Freight Ratio", "value": f"{data['freight_to_price_ratio']:.2f}", "inline": True},
            {"name": "Order Basket", "value": f"R$ {data['total_order_price']:.2f}", "inline": True},
            {"name": "Mandated Action", "value": action, "inline": False}
        ]
        dispatch_discord_alert(
            webhook_url=DISCORD_WAREHOUSE_WEBHOOK_URL,
            title="🚨 HIGH PRE-DISPATCH FRICTION DETECTED",
            description="An order has exceeded the risk boundary (> 53.97%). Priority handling required.",
            color=15158332,  # Red
            fields=fields
        )
    else:
        action = "Clear for standard automated fulfillment line."
        tier = "Standard Delivery Clearance"

    return SatisfactionOutput(
        dissatisfaction_risk=round(risk_prob, 4),
        alert_triggered=alert_triggered,
        recommended_action=action,
        risk_tier=tier
    )


# ==========================================
# 2. VIP Repeat Purchase Endpoint
# ==========================================
@app.post("/predict/repeat-purchase", response_model=RepeatPurchaseOutput)
def predict_repeat_purchase(payload: RepeatPurchaseInput):
    data = payload.dict()
    
    # Auto-calculate ratio if missing
    if not data.get("first_order_freight_ratio") or data["first_order_freight_ratio"] == 0:
        spend = data["first_order_spend"]
        freight = data["first_order_freight"]
        data["first_order_freight_ratio"] = freight / spend if spend > 0 else 0.0

    df = pd.DataFrame([data])
    
    if repeat_pipeline is not None:
        try:
            repeat_prob = float(repeat_pipeline.predict_proba(df)[0][1])
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
    else:
        repeat_prob = 0.08 if data["first_order_spend"] >= 150 else 0.02

    is_vip = repeat_prob >= VIP_THRESHOLD
    
    if is_vip:
        if data["first_order_spend"] >= 250:
            crm_action = "Enroll into Executive Tier. Dispatch an unprompted R$ 30 thank-you voucher within 72 hours."
        elif data["first_order_items_count"] > 1:
            crm_action = "Trigger category cross-sell campaign offering bundle discounts for their next checkout cycle."
        else:
            crm_action = "Dispatch targeted 10% repeat purchase coupon with priority customer support tagging."
            
        tier = "High-LTV VIP Prospect"
        
        # Trigger silent CRM webhook
        fields = [
            {"name": "Propensity Score", "value": f"{repeat_prob*100:.1f}%", "inline": True},
            {"name": "Spend Tier", "value": f"R$ {data['first_order_spend']:.2f}", "inline": True},
            {"name": "State", "value": data["customer_state"], "inline": True},
            {"name": "Targeted Action", "value": crm_action, "inline": False}
        ]
        dispatch_discord_alert(
            webhook_url=DISCORD_CRM_WEBHOOK_URL,
            title="⭐ HIGH-VALUE VIP PROSPECT IDENTIFIED",
            description="First-order retention propensity has exceeded the VIP cutoff (> 5.0%).",
            color=1752220,  # Green / Gold
            fields=fields
        )
    else:
        crm_action = "Standard promotional nurture newsletter loop."
        tier = "Standard Transactional"

    return RepeatPurchaseOutput(
        repeat_probability=round(repeat_prob, 4),
        is_high_potential=is_vip,
        crm_action=crm_action,
        customer_tier=tier
    )