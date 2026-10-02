"""
Pydantic Schemas for Olist Customer Intelligence API
===================================================
Feature contracts strictly aligned with app.py and trained LightGBM pipelines.
"""

from pydantic import BaseModel, Field
from typing import Optional


# ==========================================
# 1. Satisfaction / Friction Risk Schema
# ==========================================
class SatisfactionInput(BaseModel):
    # Features coming directly from app.py simulator
    total_order_price: float = Field(..., example=85.0, description="Total order basket value")
    total_freight_value: float = Field(..., example=45.0, description="Total freight charge")
    total_volume_cm3: float = Field(..., example=12000.0, description="Package volume in cm3")
    total_weight_g: float = Field(..., example=2500.0, description="Package weight in grams")
    estimated_transit_days: float = Field(..., example=28.0, description="Expected SLA delivery window")
    
    # Optional / defaulted features matching training schema
    freight_to_price_ratio: Optional[float] = Field(default=None, example=0.5294)
    is_cross_state: Optional[int] = Field(default=0, example=1)
    payment_installments_max: Optional[int] = Field(default=1, example=2)
    total_items_count: Optional[int] = Field(default=1, example=1)
    payment_sequential_count: Optional[int] = Field(default=1, example=1)
    purchase_year: Optional[int] = Field(default=2018, example=2018)
    purchase_month: Optional[int] = Field(default=5, example=5)
    purchase_day: Optional[int] = Field(default=15, example=15)
    purchase_dayofweek: Optional[int] = Field(default=2, example=2)
    purchase_hour: Optional[int] = Field(default=14, example=14)
    preferred_payment_type: Optional[str] = Field(default="credit_card", example="credit_card")


class SatisfactionOutput(BaseModel):
    dissatisfaction_risk: float
    alert_triggered: bool
    recommended_action: str
    risk_tier: str


# ==========================================
# 2. VIP Repeat Purchase Schema
# ==========================================
class RepeatPurchaseInput(BaseModel):
    first_order_spend: float = Field(..., example=180.0, description="First order monetary spend")
    first_order_freight: float = Field(..., example=25.0, description="First order freight amount")
    
    # Optional / defaulted features
    first_order_freight_ratio: Optional[float] = Field(default=None, example=0.1388)
    first_order_items_count: Optional[int] = Field(default=1, example=1)
    first_order_installments: Optional[int] = Field(default=1, example=2)
    first_order_payment_type: Optional[str] = Field(default="credit_card", example="credit_card")
    customer_state: Optional[str] = Field(default="SP", example="SP")
    purchase_month: Optional[int] = Field(default=5, example=5)
    purchase_hour: Optional[int] = Field(default=14, example=14)
    purchase_dayofweek: Optional[int] = Field(default=2, example=2)


class RepeatPurchaseOutput(BaseModel):
    repeat_probability: float
    is_high_potential: bool
    crm_action: str
    customer_tier: str