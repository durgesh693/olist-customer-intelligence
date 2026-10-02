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
    total_order_price: float = Field(..., example=85.0, description="Total order basket value")
    total_freight_value: float = Field(..., example=45.0, description="Total freight charge")
    freight_to_price_ratio: Optional[float] = Field(None, example=0.5294, description="Calculated freight ratio")
    total_items_count: int = Field(default=1, example=1)
    total_weight_g: float = Field(..., example=2500.0, description="Package weight in grams")
    total_volume_cm3: float = Field(..., example=12000.0, description="Package volume in cm3")
    estimated_transit_days: float = Field(..., example=28.0, description="Expected SLA delivery window")
    is_cross_state: int = Field(default=0, example=1, description="Interstate delivery flag")
    payment_sequential_count: int = Field(default=1, example=1)
    payment_installments_max: int = Field(default=1, example=2)
    purchase_year: int = Field(default=2018, example=2018)
    purchase_month: int = Field(default=5, example=5)
    purchase_day: int = Field(default=15, example=15)
    purchase_dayofweek: int = Field(default=2, example=2)
    purchase_hour: int = Field(default=14, example=14)
    preferred_payment_type: str = Field(default="credit_card", example="credit_card")


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
    first_order_freight_ratio: Optional[float] = Field(None, example=0.1388)
    first_order_items_count: int = Field(default=1, example=1)
    first_order_installments: int = Field(default=1, example=2)
    first_order_payment_type: str = Field(default="credit_card", example="credit_card")
    customer_state: str = Field(default="SP", example="SP")
    purchase_month: int = Field(default=5, example=5)
    purchase_hour: int = Field(default=14, example=14)
    purchase_dayofweek: int = Field(default=2, example=2)


class RepeatPurchaseOutput(BaseModel):
    repeat_probability: float
    is_high_potential: bool
    crm_action: str
    customer_tier: str