"""
Business ROI and Cost-Benefit Simulation Engine
===============================================
Computes projected operational savings and financial ROI resulting from
early proactive pre-dispatch interventions using the tuned LightGBM model.
"""

import os
import sys
import pandas as pd
import numpy as np

# Resolve parent directory for modular imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


class LogisticsROICalculator:
    """Simulates financial trade-offs between proactive logistics upgrades and customer churn costs."""

    def __init__(
        self,
        intervention_cost_brl: float = 12.0,   # Cost of priority dispatch / expedited carrier re-route
        dissatisfaction_cost_brl: float = 85.0, # Cost of customer support escalation, refund claims & churn
        intervention_success_rate: float = 0.65 # Likelihood of preventing a 1-star review via proactive intervention
    ):
        self.intervention_cost = intervention_cost_brl
        self.dissatisfaction_cost = dissatisfaction_cost_brl
        self.success_rate = intervention_success_rate

    def calculate_simulation(
        self,
        total_annual_orders: int = 100000,
        baseline_dissatisfaction_rate: float = 0.15,
        model_precision: float = 0.46,
        model_recall: float = 0.62
    ) -> dict:
        """
        Projects operational savings against a default baseline where no proactive alerts exist.
        """
        # Baseline without model
        actual_dissatisfied_orders = total_annual_orders * baseline_dissatisfaction_rate
        baseline_cost = actual_dissatisfied_orders * self.dissatisfaction_cost

        # Projections with model enabled
        # True Positives detected
        true_positives = actual_dissatisfied_orders * model_recall
        # Total orders flagged for alert (TP + FP) based on model precision
        total_alerts_triggered = true_positives / model_precision
        false_positives = total_alerts_triggered - true_positives

        # Intervention expenses
        total_intervention_spend = total_alerts_triggered * self.intervention_cost

        # Dissatisfaction damage prevented
        prevented_cases = true_positives * self.success_rate
        remaining_dissatisfied_cases = actual_dissatisfied_orders - prevented_cases
        residual_dissatisfaction_cost = remaining_dissatisfied_cases * self.dissatisfaction_cost

        # Total operational cost with machine learning system
        post_ml_total_cost = total_intervention_spend + residual_dissatisfaction_cost

        # Net financial benefit
        net_annual_savings = baseline_cost - post_ml_total_cost
        roi_percentage = (net_annual_savings / total_intervention_spend) * 100 if total_intervention_spend > 0 else 0.0

        return {
            "Total Annual Orders": total_annual_orders,
            "Baseline Dissatisfaction Cost (BRL)": round(baseline_cost, 2),
            "Proactive Warehouse Alerts Triggered": int(total_alerts_triggered),
            "Friction Cases Successfully Prevented": int(prevented_cases),
            "Total Intervention Cost (BRL)": round(total_intervention_spend, 2),
            "Net Annual Financial Savings (BRL)": round(net_annual_savings, 2),
            "Operational ROI (%)": round(roi_percentage, 1)
        }

    def export_summary_table(self, output_path: str = None) -> pd.DataFrame:
        """
        Simulates multiple order volume scales and saves summary table.
        """
        scenarios = [50000, 100000, 250000]
        records = [self.calculate_simulation(total_annual_orders=vol) for vol in scenarios]
        df = pd.DataFrame(records)

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df.to_csv(output_path, index=False)
            print(f"[EXPORTED] ROI simulation table saved to: {output_path}")

        return df


if __name__ == "__main__":
    calc = LogisticsROICalculator()
    results = calc.calculate_simulation()

    print("=" * 60)
    print("      EXECUTIVE BUSINESS ROI PROJECTION SUMMARY")
    print("=" * 60)
    for key, value in results.items():
        if "BRL" in key:
            print(f"{key:<40}: R$ {value:,.2f}")
        elif "%" in key:
            print(f"{key:<40}: {value}%")
        else:
            print(f"{key:<40}: {value:,}")
    print("=" * 60)

    # Save to reports
    csv_out = os.path.join(PROJECT_ROOT, "reports", "business_roi_projection.csv")
    calc.export_summary_table(output_path=csv_out)