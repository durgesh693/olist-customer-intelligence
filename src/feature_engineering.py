"""
Feature Engineering Pipelines
=============================
Contains pure transformations for:
1. RFM (Recency, Frequency, Monetary) Customer Aggregations (Track 2)
2. Pre-Dispatch Order Friction Features (Track 1)
"""

import pandas as pd
import numpy as np


class CustomerRFMBuilder:
    """Computes Recency, Frequency, and Monetary metrics for unique customers."""

    @staticmethod
    def build_rfm_table(orders_df: pd.DataFrame, 
                        items_df: pd.DataFrame, 
                        customers_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates customer-level RFM metrics using delivered orders.
        """
        # 1. Filter delivered orders only
        delivered_orders = orders_df[orders_df["order_status"] == "delivered"].copy()
        delivered_orders["order_purchase_timestamp"] = pd.to_datetime(delivered_orders["order_purchase_timestamp"])

        # 2. Reference point: Max date in dataset + 1 day
        snapshot_date = delivered_orders["order_purchase_timestamp"].max() + pd.Timedelta(days=1)

        # 3. Monetary calculation (Item price + Freight value)
        item_spend = items_df.groupby("order_id").agg({
            "price": "sum",
            "freight_value": "sum"
        }).reset_index()
        item_spend["order_total_spend"] = item_spend["price"] + item_spend["freight_value"]

        # 4. Merge orders with customer and spend data
        merged = delivered_orders.merge(customers_df, on="customer_id", how="inner")
        merged = merged.merge(item_spend[["order_id", "order_total_spend"]], on="order_id", how="inner")

        # 5. Group by customer_unique_id
        rfm = merged.groupby("customer_unique_id").agg(
            recency=("order_purchase_timestamp", lambda dates: (snapshot_date - dates.max()).days),
            frequency=("order_id", "nunique"),
            monetary=("order_total_spend", "sum")
        ).reset_index()

        # Handle zero or negative monetary edge-cases
        rfm["monetary"] = rfm["monetary"].clip(lower=0.01)
        
        return rfm


class PreDispatchFeatureBuilder:
    """Transforms raw order, item, and payment data into pre-dispatch tabular features."""

    @staticmethod
    def extract_features(orders_df: pd.DataFrame,
                         items_df: pd.DataFrame,
                         payments_df: pd.DataFrame,
                         customers_df: pd.DataFrame,
                         sellers_df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates clean order-level training features strictly known BEFORE order delivery.
        """
        # 1. Aggregate Order Items (Count, Total Price, Total Freight, Total Weight, Volume)
        items = items_df.copy()
        items["volume_cm3"] = (
            items["product_length_cm"].fillna(1) * 
            items["product_height_cm"].fillna(1) * 
            items["product_width_cm"].fillna(1)
        )
        
        item_agg = items.groupby("order_id").agg(
            total_items_count=("order_item_id", "count"),
            total_order_price=("price", "sum"),
            total_freight_value=("freight_value", "sum"),
            total_weight_g=("product_weight_g", "sum"),
            total_volume_cm3=("volume_cm3", "sum"),
            first_seller_id=("seller_id", "first")
        ).reset_index()

        # Mathematical ratio
        item_agg["freight_to_price_ratio"] = (
            item_agg["total_freight_value"] / item_agg["total_order_price"]
        ).fillna(0.0)

        # 2. Aggregate Payments (Preferred type, Max installments)
        pay_agg = payments_df.groupby("order_id").agg(
            preferred_payment_type=("payment_type", lambda x: x.mode()[0] if not x.empty else "credit_card"),
            payment_installments_max=("payment_installments", "max"),
            payment_sequential_count=("payment_sequential", "max")
        ).reset_index()

        # 3. Merge base orders
        df = orders_df[["order_id", "customer_id", "order_purchase_timestamp", "order_estimated_delivery_date"]].copy()
        df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
        df["order_estimated_delivery_date"] = pd.to_datetime(df["order_estimated_delivery_date"])

        # SLA estimated days
        df["estimated_transit_days"] = (
            df["order_estimated_delivery_date"] - df["order_purchase_timestamp"]
        ).dt.total_seconds() / 86400.0

        # Temporal breakdown
        df["purchase_year"] = df["order_purchase_timestamp"].dt.year
        df["purchase_month"] = df["order_purchase_timestamp"].dt.month
        df["purchase_day"] = df["order_purchase_timestamp"].dt.day
        df["purchase_dayofweek"] = df["order_purchase_timestamp"].dt.dayofweek
        df["purchase_hour"] = df["order_purchase_timestamp"].dt.hour

        # 4. Join datasets
        df = df.merge(item_agg, on="order_id", how="inner")
        df = df.merge(pay_agg, on="order_id", how="inner")
        df = df.merge(customers_df[["customer_id", "customer_state"]], on="customer_id", how="inner")
        df = df.merge(sellers_df[["seller_id", "seller_state"]], left_on="first_seller_id", right_on="seller_id", how="left")

        # Cross-state indicator
        df["is_cross_state"] = (df["customer_state"] != df["seller_state"]).astype(int)

        # Drop temporary joining columns
        df = df.drop(columns=["seller_id", "first_seller_id", "customer_id", 
                              "order_purchase_timestamp", "order_estimated_delivery_date"])

        return df