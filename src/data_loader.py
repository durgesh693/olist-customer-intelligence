"""
Data Ingestion and Loading Pipeline
==================================
Handles finding raw Olist CSV datasets, basic sanitization,
and returning cleaned pandas DataFrames.
"""

import os
import pandas as pd
from typing import Dict, Optional

# Project root path resolver
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")


class OlistDataLoader:
    """Loads and standardizes raw Olist e-commerce CSV files."""

    FILE_MAPPING = {
        "customers": "olist_customers_dataset.csv",
        "orders": "olist_orders_dataset.csv",
        "items": "olist_order_items_dataset.csv",
        "payments": "olist_order_payments_dataset.csv",
        "reviews": "olist_order_reviews_dataset.csv",
        "products": "olist_products_dataset.csv",
        "sellers": "olist_sellers_dataset.csv",
        "geolocation": "olist_geolocation_dataset.csv",
        "category_translation": "product_category_name_translation.csv"
    }

    def __init__(self, raw_data_dir: Optional[str] = None):
        self.raw_data_dir = raw_data_dir or DATA_RAW_DIR

    def load_single_dataset(self, dataset_key: str) -> pd.DataFrame:
        """Ek specific CSV dataset ko read karta hai."""
        if dataset_key not in self.FILE_MAPPING:
            raise KeyError(f"Unknown dataset '{dataset_key}'. Available: {list(self.FILE_MAPPING.keys())}")

        file_name = self.FILE_MAPPING[dataset_key]
        file_path = os.path.join(self.raw_data_dir, file_name)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}. Please check data/raw/ directory.")

        return pd.read_csv(file_path)

    def load_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """Saare raw CSV files ko ek dictionary mein load karta hai."""
        datasets = {}
        for key in self.FILE_MAPPING:
            try:
                datasets[key] = self.load_single_dataset(key)
                print(f"[LOADED] {key} shape: {datasets[key].shape}")
            except FileNotFoundError:
                print(f"[SKIP] {key} file not present in {self.raw_data_dir}")
        return datasets

    @staticmethod
    def parse_order_dates(orders_df: pd.DataFrame) -> pd.DataFrame:
        """Orders table ke timestamp columns ko datetime format mein convert karta hai."""
        df = orders_df.copy()
        date_cols = [
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "order_estimated_delivery_date"
        ]
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        return df


if __name__ == "__main__":
    # Test script directly
    loader = OlistDataLoader()
    try:
        orders = loader.load_single_dataset("orders")
        orders = OlistDataLoader.parse_order_dates(orders)
        print(f"Data loader successfully verified. Loaded orders count: {len(orders)}")
    except Exception as e:
        print(f"Test run note: {e}")