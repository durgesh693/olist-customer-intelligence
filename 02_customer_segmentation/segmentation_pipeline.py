"""
Customer Segmentation Pipeline Module
=====================================
Executes RFM feature transformations, evaluates optimal clusters
via Elbow Method and Silhouette analysis, trains K-Means,
and generates customer persona segments and visualization artifacts.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Resolve root directory for modular imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.data_loader import OlistDataLoader
from src.feature_engineering import CustomerRFMBuilder


class CustomerSegmentationPipeline:
    """Manages scaling, clustering, and profiling of customer RFM data."""

    def __init__(self, n_clusters: int = 4, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.kmeans_model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)

    def load_and_transform_features(self) -> tuple[pd.DataFrame, np.ndarray]:
        """
        Loads raw datasets, derives RFM metrics, and applies log transformation and scaling.
        """
        loader = OlistDataLoader()
        orders = loader.load_single_dataset("orders")
        items = loader.load_single_dataset("items")
        customers = loader.load_single_dataset("customers")

        # Generate base RFM table using src builder
        rfm_df = CustomerRFMBuilder.build_rfm_table(orders, items, customers)

        # Apply Log transformation to mitigate extreme e-commerce skewness
        rfm_log = pd.DataFrame()
        rfm_log["recency"] = np.log1p(rfm_df["recency"])
        rfm_log["frequency"] = np.log1p(rfm_df["frequency"])
        rfm_log["monetary"] = np.log1p(rfm_df["monetary"])

        # Standardize features
        scaled_features = self.scaler.fit_transform(rfm_log)

        return rfm_df, scaled_features

    def evaluate_optimal_clusters(self, scaled_features: np.ndarray, max_k: int = 8, output_path: str = None) -> None:
        """
        Calculates inertia and silhouette scores across multiple K values and plots the elbow curve.
        """
        inertias = []
        silhouette_scores = []
        k_range = range(2, max_k + 1)

        # Subsample for faster silhouette evaluation on large datasets
        sample_size = min(15000, len(scaled_features))
        np.random.seed(self.random_state)
        sample_indices = np.random.choice(len(scaled_features), size=sample_size, replace=False)
        sub_features = scaled_features[sample_indices]

        for k in k_range:
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init=5)
            labels = km.fit_predict(scaled_features)
            inertias.append(km.inertia_)
            sub_labels = labels[sample_indices]
            silhouette_scores.append(silhouette_score(sub_features, sub_labels))

        # Plot evaluation curves
        fig, ax1 = plt.subplots(figsize=(10, 5))

        ax1.set_xlabel("Number of Clusters (k)", fontsize=11)
        ax1.set_ylabel("Inertia (Elbow Sum of Squares)", color="tab:blue", fontsize=11)
        ax1.plot(k_range, inertias, marker="o", color="tab:blue", linewidth=2, label="Inertia")
        ax1.tick_params(axis="y", labelcolor="tab:blue")

        ax2 = ax1.twinx()
        ax2.set_ylabel("Silhouette Score", color="tab:orange", fontsize=11)
        ax2.plot(k_range, silhouette_scores, marker="s", color="tab:orange", linewidth=2, linestyle="--", label="Silhouette")
        ax2.tick_params(axis="y", labelcolor="tab:orange")

        plt.title("Cluster Evaluation: Elbow Method vs. Silhouette Score", fontsize=13, pad=15)
        fig.tight_layout()

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300)
            print(f"[SAVED] Cluster evaluation chart saved to: {output_path}")
        plt.close()

    def run_clustering_and_labeling(self) -> pd.DataFrame:
        """
        Fits K-Means, maps cluster IDs to human-interpretable business personas,
        and returns the finalized dataframe.
        """
        rfm_df, scaled_features = self.load_and_transform_features()

        # Fit model and predict clusters
        cluster_labels = self.kmeans_model.fit_predict(scaled_features)
        rfm_df["cluster"] = cluster_labels

        # Calculate cluster medians to dynamically map business persona titles
        cluster_stats = rfm_df.groupby("cluster").agg({
            "recency": "median",
            "frequency": "median",
            "monetary": "median"
        }).reset_index()

        def assign_persona(row):
            if row["frequency"] > 1.0:
                return "Loyal Repeat Buyers"
            elif row["recency"] <= rfm_df["recency"].median() and row["monetary"] >= rfm_df["monetary"].median():
                return "Recent Fresh Buyers"
            elif row["recency"] > rfm_df["recency"].median() and row["monetary"] >= rfm_df["monetary"].median():
                return "High-Value Dormant"
            else:
                return "Low-Value Lost"

        cluster_persona_mapping = {}
        for _, row in cluster_stats.iterrows():
            cluster_persona_mapping[int(row["cluster"])] = assign_persona(row)

        rfm_df["segment_name"] = rfm_df["cluster"].map(cluster_persona_mapping)

        return rfm_df

    def plot_segment_profiles(self, rfm_df: pd.DataFrame, output_path: str = None) -> None:
        """
        Generates boxplots depicting Recency, Frequency, and Monetary distribution per segment.
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        sns.boxplot(data=rfm_df, x="segment_name", y="recency", ax=axes[0], palette="Set2")
        axes[0].set_title("Recency Distribution (Days)", fontsize=12)
        axes[0].set_xlabel("")
        axes[0].tick_params(axis="x", rotation=25)

        sns.boxplot(data=rfm_df, x="segment_name", y="frequency", ax=axes[1], palette="Set2")
        axes[1].set_title("Frequency Distribution (Orders Count)", fontsize=12)
        axes[1].set_xlabel("")
        axes[1].tick_params(axis="x", rotation=25)

        sns.boxplot(data=rfm_df, x="segment_name", y="monetary", ax=axes[2], palette="Set2")
        axes[2].set_title("Monetary Spend (R$ - Log Scale)", fontsize=12)
        axes[2].set_yscale("log")
        axes[2].set_xlabel("")
        axes[2].tick_params(axis="x", rotation=25)

        plt.suptitle("Customer Segment Behavioral Profiles", fontsize=14, y=1.03)
        plt.tight_layout()

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300)
            print(f"[SAVED] Segment profile boxplot saved to: {output_path}")
        plt.close()


if __name__ == "__main__":
    pipeline = CustomerSegmentationPipeline(n_clusters=4)

    # 1. Feature preparation & evaluation plot
    print("[1/3] Loading data and evaluating optimal clusters...")
    _, scaled_data = pipeline.load_and_transform_features()
    eval_chart_path = os.path.join(PROJECT_ROOT, "reports", "figures", "kmeans_elbow_evaluation.png")
    pipeline.evaluate_optimal_clusters(scaled_data, max_k=7, output_path=eval_chart_path)

    # 2. Execute clustering and persona labeling
    print("[2/3] Fitting K-Means and mapping business personas...")
    segmented_customers = pipeline.run_clustering_and_labeling()

    # 3. Export CSV artifact used by Streamlit Dashboard
    csv_export_path = os.path.join(PROJECT_ROOT, "data", "processed", "customer_rfm_segments.csv")
    os.makedirs(os.path.dirname(csv_export_path), exist_ok=True)
    segmented_customers.to_csv(csv_export_path, index=False)
    print(f"[SAVED] Processed RFM dataset exported to: {csv_export_path}")

    # 4. Generate profile visual artifact
    print("[3/3] Generating segment profile visualizations...")
    profile_chart_path = os.path.join(PROJECT_ROOT, "reports", "figures", "customer_segments_profile.png")
    pipeline.plot_segment_profiles(segmented_customers, output_path=profile_chart_path)
    print("✅ Customer segmentation pipeline execution completed successfully!")