"""
Geographic Customer and Logistics Analytics Module
==================================================
Analyzes order volume, average spend, and delivery transit latency
across different Brazilian states to pinpoint regional friction.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.data_loader import OlistDataLoader


class GeographicAnalytics:
    """Computes regional performance metrics and delivery SLA variances."""

    def __init__(self, data_loader: OlistDataLoader):
        self.loader = data_loader

    def compute_state_performance(self) -> pd.DataFrame:
        """
        Merges orders, items, and customers to calculate metrics aggregated by state.
        """
        orders = self.loader.load_single_dataset("orders")
        items = self.loader.load_single_dataset("items")
        customers = self.loader.load_single_dataset("customers")

        # Parse delivery dates
        orders = orders[orders["order_status"] == "delivered"].copy()
        orders["order_purchase_timestamp"] = pd.to_datetime(orders["order_purchase_timestamp"])
        orders["order_delivered_customer_date"] = pd.to_datetime(orders["order_delivered_customer_date"])
        orders["order_estimated_delivery_date"] = pd.to_datetime(orders["order_estimated_delivery_date"])

        # Actual transit days
        orders["actual_delivery_days"] = (
            orders["order_delivered_customer_date"] - orders["order_purchase_timestamp"]
        ).dt.total_seconds() / 86400.0

        # Delay indicator (1 if delivered after estimated date)
        orders["is_delayed"] = (
            orders["order_delivered_customer_date"] > orders["order_estimated_delivery_date"]
        ).astype(int)

        # Monetary totals per order
        order_totals = items.groupby("order_id").agg(
            order_spend=("price", "sum"),
            order_freight=("freight_value", "sum")
        ).reset_index()

        # Merge all components
        df = orders.merge(customers[["customer_id", "customer_state"]], on="customer_id", how="inner")
        df = df.merge(order_totals, on="order_id", how="inner")

        # State-level aggregation
        state_summary = df.groupby("customer_state").agg(
            total_orders=("order_id", "count"),
            avg_order_value=("order_spend", "mean"),
            avg_freight_value=("order_freight", "mean"),
            avg_delivery_days=("actual_delivery_days", "mean"),
            delay_rate_pct=("is_delayed", lambda x: (x.sum() / len(x)) * 100)
        ).reset_index()

        state_summary["freight_share_pct"] = (
            state_summary["avg_freight_value"] / state_summary["avg_order_value"] * 100
        )

        return state_summary.sort_values(by="total_orders", ascending=False)

    def export_summary(self, csv_output_path: str) -> pd.DataFrame:
        """
        Executes aggregation and exports CSV summary table.
        """
        summary = self.compute_state_performance()
        os.makedirs(os.path.dirname(csv_output_path), exist_ok=True)
        summary.to_csv(csv_output_path, index=False)
        print(f"[EXPORTED] Regional metrics saved to: {csv_output_path}")
        return summary


if __name__ == "__main__":
    loader = OlistDataLoader()
    geo = GeographicAnalytics(loader)
    
    csv_file = os.path.join(PROJECT_ROOT, "reports", "geographic_performance_summary.csv")
    geo.export_summary(csv_file)