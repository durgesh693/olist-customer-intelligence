"""
Repeat Purchase Dataset Builder Module
======================================
Constructs customer-level feature matrices derived strictly from the customer's 
first completed transaction to prevent future data leakage.
"""

import os
import sys
import pandas as pd
import numpy as np

# Resolve parent directory to access src package
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.data_loader import OlistDataLoader


class RepeatPurchaseDatasetBuilder:
    """Builds pre-dispatch first-order feature matrix with binary repurchase labels."""

    def __init__(self, data_loader: OlistDataLoader):
        self.loader = data_loader

    def build_dataset(self) -> pd.DataFrame:
        """
        Extracts first-order attributes and constructs the binary target variable
        (1 if customer made > 1 lifetime orders, else 0).
        """
        # Load raw relational datasets
        orders = self.loader.load_single_dataset("orders")
        items = self.loader.load_single_dataset("items")
        payments = self.loader.load_single_dataset("payments")
        customers = self.loader.load_single_dataset("customers")

        # Filter delivered transactions and standardize timestamps
        orders_delivered = orders[orders["order_status"] == "delivered"].copy()
        orders_delivered["order_purchase_timestamp"] = pd.to_datetime(
            orders_delivered["order_purchase_timestamp"]
        )

        # Merge with customer identification table
        customer_orders = orders_delivered.merge(
            customers[["customer_id", "customer_unique_id", "customer_state"]],
            on="customer_id",
            how="inner"
        )

        # Construct target variable: lifetime transaction count > 1
        total_orders = (
            customer_orders.groupby("customer_unique_id")["order_id"]
            .nunique()
            .rename("total_lifetime_orders")
        )
        customer_orders = customer_orders.merge(total_orders, on="customer_unique_id", how="left")
        customer_orders["is_repeat_buyer"] = (customer_orders["total_lifetime_orders"] > 1).astype(int)

        # Sort chronologically to isolate the true initial transaction
        sorted_orders = customer_orders.sort_values(
            by=["customer_unique_id", "order_purchase_timestamp"]
        )
        first_orders = sorted_orders.groupby("customer_unique_id").first().reset_index()

        # Compute item aggregates for the initial purchase
        item_features = items.groupby("order_id").agg(
            first_order_items_count=("order_item_id", "count"),
            first_order_spend=("price", "sum"),
            first_order_freight=("freight_value", "sum")
        ).reset_index()

        item_features["first_order_freight_ratio"] = (
            item_features["first_order_freight"] / item_features["first_order_spend"]
        ).fillna(0.0)

        # Compute payment aggregates for the initial purchase
        payment_features = payments.groupby("order_id").agg(
            first_order_payment_type=("payment_type", lambda s: s.mode()[0] if not s.empty else "credit_card"),
            first_order_installments=("payment_installments", "max")
        ).reset_index()

        # Merge aggregates to first order table
        first_order_df = first_orders.merge(item_features, on="order_id", how="inner")
        first_order_df = first_order_df.merge(payment_features, on="order_id", how="inner")

        # Extract temporal predictors
        first_order_df["purchase_month"] = first_order_df["order_purchase_timestamp"].dt.month
        first_order_df["purchase_hour"] = first_order_df["order_purchase_timestamp"].dt.hour
        first_order_df["purchase_dayofweek"] = first_order_df["order_purchase_timestamp"].dt.dayofweek

        # Retain finalized training features
        features_df = first_order_df[[
            "customer_unique_id",
            "first_order_spend",
            "first_order_freight",
            "first_order_freight_ratio",
            "first_order_items_count",
            "first_order_installments",
            "first_order_payment_type",
            "customer_state",
            "purchase_month",
            "purchase_hour",
            "purchase_dayofweek",
            "is_repeat_buyer"
        ]].copy()

        return features_df


if __name__ == "__main__":
    loader = OlistDataLoader()
    builder = RepeatPurchaseDatasetBuilder(loader)
    dataset = builder.build_dataset()
    print(f"[SUCCESS] Dataset built successfully with {len(dataset)} records.")
    print(f"Repeat Buyers: {dataset['is_repeat_buyer'].sum()} ({dataset['is_repeat_buyer'].mean()*100:.2f}%)")