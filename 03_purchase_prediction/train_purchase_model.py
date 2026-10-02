"""
Repeat Purchase Classifier Training Pipeline
============================================
Executes end-to-end training of the repeat buyer prediction pipeline.
Applies class-weight balancing, evaluates discrimination curves,
and exports production artifacts.
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
    classification_report
)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.data_loader import OlistDataLoader
from dataset_builder import RepeatPurchaseDatasetBuilder


def run_training_pipeline():
    """Executes data preparation, model training, evaluation, and artifact export."""
    print("=" * 60)
    print("Step 1: Extracting leak-free first-order features...")
    print("=" * 60)
    
    loader = OlistDataLoader()
    builder = RepeatPurchaseDatasetBuilder(loader)
    dataset = builder.build_dataset()

    X = dataset.drop(columns=["customer_unique_id", "is_repeat_buyer"])
    y = dataset["is_repeat_buyer"]

    # Stratified 80/20 train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    numeric_features = [
        "first_order_spend",
        "first_order_freight",
        "first_order_freight_ratio",
        "first_order_items_count",
        "first_order_installments",
        "purchase_month",
        "purchase_hour",
        "purchase_dayofweek"
    ]
    categorical_features = ["first_order_payment_type", "customer_state"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", RobustScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
        ]
    )

    # Compute scale_pos_weight for imbalance adjustment (~97:3 ratio)
    pos_weight = float((len(y_train) - y_train.sum()) / y_train.sum())

    classifier = LGBMClassifier(
        n_estimators=150,
        learning_rate=0.03,
        max_depth=5,
        scale_pos_weight=pos_weight,
        random_state=42,
        verbosity=-1
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    print("\nStep 2: Fitting LightGBM pipeline with class balancing...")
    pipeline.fit(X_train, y_train)

    print("\nStep 3: Evaluating performance metrics on unseen holdout test set...")
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)

    print("-" * 40)
    print(f"Holdout ROC-AUC Score : {roc_auc:.4f}")
    print(f"Holdout PR-AUC Score  : {pr_auc:.4f} (Baseline: {y_test.mean():.4f})")
    print("-" * 40)

    # Export validation curve visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    axes[0].plot(fpr, tpr, color="#2980b9", lw=2, label=f"LGBM (AUC = {roc_auc:.3f})")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[0].set_title("ROC Curve (Repeat Buyer Prediction)", fontsize=12)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend(loc="lower right")

    # PR Curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    axes[1].plot(recall, precision, color="#27ae60", lw=2, label=f"PR Curve (AUC = {pr_auc:.3f})")
    axes[1].axhline(y=y_test.mean(), linestyle="--", color="gray", label=f"Baseline ({y_test.mean():.3f})")
    axes[1].set_title("Precision-Recall Curve (Imbalanced Cohort)", fontsize=12)
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(loc="upper right")

    plt.tight_layout()

    fig_output_dir = os.path.join(PROJECT_ROOT, "reports", "figures")
    os.makedirs(fig_output_dir, exist_ok=True)
    fig_path = os.path.join(fig_output_dir, "repeat_purchase_validation_curves.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[SAVED] Performance charts exported to: {fig_path}")

    # Step 4: Serialize fitted model pipeline
    print("\nStep 4: Serializing fitted pipeline artifact...")
    models_dir = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "lgbm_repeat_purchase_pipeline.pkl")
    joblib.dump(pipeline, model_path)
    print(f"[EXPORTED] Production model artifact saved to: {model_path}")
    print("=" * 60)
    print("✅ Repeat Purchase Training Pipeline finalized successfully!")
    print("=" * 60)


if __name__ == "__main__":
    run_training_pipeline()