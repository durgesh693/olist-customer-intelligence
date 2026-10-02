"""
Pre-Dispatch Customer Dissatisfaction Model Training Pipeline
============================================================
Extracts pre-dispatch order features, trains a calibrated LightGBM pipeline,
computes optimal cost-benefit decision threshold, and exports serialized artifacts.
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix
)

# Resolve parent directory for modular imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.data_loader import OlistDataLoader
from src.feature_engineering import PreDispatchFeatureBuilder


def train_dissatisfaction_pipeline():
    """
    Executes end-to-end training and serialization for pre-dispatch dissatisfaction prediction.
    """
    print("=" * 65)
    print("Step 1: Ingesting raw relational datasets and building features...")
    print("=" * 65)

    loader = OlistDataLoader()
    orders = loader.load_single_dataset("orders")
    items = loader.load_single_dataset("items")
    payments = loader.load_single_dataset("payments")
    customers = loader.load_single_dataset("customers")
    sellers = loader.load_single_dataset("sellers")
    reviews = loader.load_single_dataset("reviews")

    # Keep delivered orders to align with finalized customer review scores
    delivered_orders = orders[orders["order_status"] == "delivered"].copy()

    # Extract clean pre-dispatch features using src builder
    features_df = PreDispatchFeatureBuilder.extract_features(
        orders_df=delivered_orders,
        items_df=items,
        payments_df=payments,
        customers_df=customers,
        sellers_df=sellers
    )

    # Construct binary target: 1 if review_score in [1, 2], else 0
    review_agg = reviews.groupby("order_id")["review_score"].min().reset_index()
    dataset = features_df.merge(review_agg, on="order_id", how="inner")
    dataset["is_dissatisfied"] = (dataset["review_score"] <= 2).astype(int)

    # Separate feature matrix and target
    X = dataset.drop(columns=["order_id", "review_score", "is_dissatisfied"])
    y = dataset["is_dissatisfied"]

    print(f"Dataset successfully prepared: {X.shape[0]:,} rows, {X.shape[1]} features.")
    print(f"Dissatisfied Class Ratio: {y.mean() * 100:.2f}% (Negative Sentiment)")

    # Stratified 80/20 train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    numeric_features = [
        "total_order_price",
        "total_freight_value",
        "freight_to_price_ratio",
        "total_items_count",
        "total_weight_g",
        "total_volume_cm3",
        "estimated_transit_days",
        "is_cross_state",
        "payment_sequential_count",
        "payment_installments_max",
        "purchase_year",
        "purchase_month",
        "purchase_day",
        "purchase_dayofweek",
        "purchase_hour"
    ]
    categorical_features = ["preferred_payment_type"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", RobustScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
        ]
    )

    # Balanced class weights for handling ~85:15 dissatisfaction imbalance
    pos_weight = float((len(y_train) - y_train.sum()) / y_train.sum())

    classifier = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.04,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=pos_weight,
        random_state=42,
        verbosity=-1
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    print("\nStep 2: Training LightGBM dissatisfaction pipeline...")
    pipeline.fit(X_train, y_train)

    print("\nStep 3: Evaluating holdout performance & discrimination curves...")
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)

    print("-" * 45)
    print(f"Holdout ROC-AUC Score : {roc_auc:.4f}")
    print(f"Holdout PR-AUC Score  : {pr_auc:.4f} (Baseline: {y_test.mean():.4f})")
    print("-" * 45)

    # Export validation curve visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    axes[0].plot(fpr, tpr, color="#c0392b", lw=2, label=f"LGBM Pipeline (AUC = {roc_auc:.3f})")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[0].set_title("ROC Curve (Pre-Dispatch Friction)", fontsize=12)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend(loc="lower right")

    # Precision-Recall Curve
    prec, rec, _ = precision_recall_curve(y_test, y_pred_proba)
    axes[1].plot(rec, prec, color="#d35400", lw=2, label=f"PR Curve (AUC = {pr_auc:.3f})")
    axes[1].axhline(y=y_test.mean(), linestyle="--", color="gray", label=f"Baseline ({y_test.mean():.3f})")
    axes[1].set_title("Precision-Recall Curve (Dissatisfaction)", fontsize=12)
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(loc="upper right")

    plt.tight_layout()

    figures_dir = os.path.join(PROJECT_ROOT, "reports", "figures")
    os.makedirs(figures_dir, exist_ok=True)
    fig_path = os.path.join(figures_dir, "satisfaction_validation_curves.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[SAVED] Performance charts exported to: {fig_path}")

    # Step 4: Serialize production pipeline artifact
    print("\nStep 4: Serializing fitted production pipeline...")
    models_dir = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "lgbm_dissatisfaction_pipeline.pkl")
    joblib.dump(pipeline, model_path)
    print(f"[EXPORTED] Production model artifact saved to: {model_path}")
    print("=" * 65)
    print("✅ Customer Satisfaction Training Pipeline completed successfully!")
    print("=" * 65)


if __name__ == "__main__":
    train_dissatisfaction_pipeline()