"""
Cohort Retention Analysis Module
================================
Calculates monthly acquisition cohorts and computes retention matrices
to evaluate customer lifecycle durability across the platform.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to sys.path for modular imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.data_loader import OlistDataLoader


class CohortRetentionAnalyzer:
    """Computes monthly customer retention rates and generates cohort heatmaps."""

    def __init__(self, data_loader: OlistDataLoader):
        self.loader = data_loader

    def prepare_cohort_data(self) -> pd.DataFrame:
        """
        Loads orders and customer mappings to assign cohort dates and purchase periods.
        """
        orders = self.loader.load_single_dataset("orders")
        customers = self.loader.load_single_dataset("customers")

        # Keep delivered orders only to measure completed transaction retention
        orders = orders[orders["order_status"] == "delivered"].copy()
        orders["order_purchase_timestamp"] = pd.to_datetime(orders["order_purchase_timestamp"])

        # Join to get customer_unique_id
        df = orders.merge(customers[["customer_id", "customer_unique_id"]], on="customer_id", how="inner")

        # Extract purchase month (period)
        df["order_month"] = df["order_purchase_timestamp"].dt.to_period("M")

        # Assign initial cohort month per unique customer
        df["cohort_month"] = df.groupby("customer_unique_id")["order_purchase_timestamp"].transform("min").dt.to_period("M")

        return df

    @staticmethod
    def calculate_cohort_index(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the integer index representing the month offset from initial acquisition.
        """
        data = df.copy()
        year_diff = data["order_month"].dt.year - data["cohort_month"].dt.year
        month_diff = data["order_month"].dt.month - data["cohort_month"].dt.month
        data["cohort_index"] = year_diff * 12 + month_diff
        return data

    def generate_retention_matrix(self, min_orders_threshold: int = 50) -> pd.DataFrame:
        """
        Builds a percentage retention matrix with cohorts as rows and months as columns.
        """
        df = self.prepare_cohort_data()
        df = self.calculate_cohort_index(df)

        # Aggregate unique users per cohort and period
        cohort_group = df.groupby(["cohort_month", "cohort_index"])["customer_unique_id"].nunique().reset_index()
        cohort_pivot = cohort_group.pivot(index="cohort_month", columns="cohort_index", values="customer_unique_id")

        # Filter out early unstable cohorts with very few acquisitions
        cohort_sizes = cohort_pivot.iloc[:, 0]
        cohort_pivot = cohort_pivot[cohort_sizes >= min_orders_threshold]

        # Calculate percentage retention
        retention_matrix = cohort_pivot.divide(cohort_sizes, axis=0) * 100
        return retention_matrix

    def plot_retention_heatmap(self, output_path: str = None) -> None:
        """
        Plots and saves a professional retention heatmap.
        """
        matrix = self.generate_retention_matrix()

        # Limit to first 12 months for visual clarity
        plot_matrix = matrix.iloc[:, :13]

        plt.figure(figsize=(14, 8))
        sns.heatmap(
            plot_matrix,
            annot=True,
            fmt=".1f",
            cmap="Blues",
            cbar_kws={"label": "Retention Rate (%)"},
            linewidths=0.5
        )
        plt.title("Olist Monthly Customer Retention Cohorts (%)", fontsize=14, pad=15)
        plt.xlabel("Months Since First Purchase (Cohort Index)", fontsize=11)
        plt.ylabel("Acquisition Cohort", fontsize=11)
        plt.tight_layout()

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300)
            print(f"[SAVED] Retention heatmap exported to: {output_path}")
        else:
            plt.show()
        plt.close()


if __name__ == "__main__":
    loader = OlistDataLoader()
    analyzer = CohortRetentionAnalyzer(loader)
    
    # Save chart to reports/figures/
    export_file = os.path.join(PROJECT_ROOT, "reports", "figures", "cohort_retention_heatmap.png")
    analyzer.plot_retention_heatmap(output_path=export_file)