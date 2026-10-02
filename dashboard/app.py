"""
Olist Customer Intelligence - Executive Decision & Analytics Hub
================================================================
Aligns directly with trained LightGBM feature importance rankings.
Provides dynamic root-cause diagnosis and actionable operational playbooks.
"""

import os
import requests
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ==========================================
# 1. Page Configuration & Modern Theme CSS
# ==========================================
st.set_page_config(
    page_title="Olist Intelligence Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    * { font-family: 'Plus Jakarta Sans', sans-serif; }
    
    .main { background-color: #0d1117; }
    
    h1 {
        font-size: 2.5rem !important;
        font-weight: 800 !important;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem !important;
    }
    h2 { font-size: 1.8rem !important; font-weight: 700 !important; color: #f1f5f9 !important; margin-top: 1rem !important; }
    h3 { font-size: 1.3rem !important; font-weight: 600 !important; color: #94a3b8 !important; }
    
    .kpi-container {
        background: #161b22; border: 1px solid #30363d; border-radius: 12px;
        padding: 20px; text-align: center; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    .kpi-title { font-size: 0.92rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600; }
    .kpi-value { font-size: 2.1rem; font-weight: 800; color: #f0f6fc; margin: 6px 0; }
    .kpi-sub { font-size: 0.82rem; color: #38bdf8; }
    
    .recommendation-card {
        background-color: #161b22; border-radius: 10px; padding: 18px;
        border: 1px solid #30363d; margin-top: 12px;
    }
    .driver-tag {
        display: inline-block; background-color: #21262d; color: #f59e0b;
        border: 1px solid #d97706; padding: 3px 10px; border-radius: 16px;
        font-size: 0.85rem; font-weight: 600; margin-bottom: 8px;
    }
    .event-badge {
        display: inline-flex; align-items: center; gap: 6px;
        background-color: #21262d; color: #58a6ff; border: 1px solid #30363d;
        padding: 4px 12px; border-radius: 20px; font-size: 0.82rem;
        font-weight: 600; margin-top: 10px;
    }
    
    .stButton>button {
        background: linear-gradient(90deg, #2563eb, #1d4ed8);
        color: white; border: none; border-radius: 8px;
        padding: 12px 24px; font-size: 1.05rem; font-weight: 700;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.4);
    }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(37, 99, 235, 0.6); }
    </style>
""", unsafe_allow_html=True)

# Environment Endpoints
SATISFACTION_API_URL = os.getenv("SATISFACTION_API_URL", "http://localhost:8000/predict/satisfaction")
REPEAT_PURCHASE_API_URL = os.getenv("REPEAT_PURCHASE_API_URL", "http://localhost:8000/predict/repeat-purchase")

# ==========================================
# SIDEBAR: Metadata
# ==========================================
st.sidebar.markdown("## ⚡ Olist Operations Hub")
st.sidebar.caption("Dual-Pipeline ML System")
st.sidebar.markdown("---")

st.sidebar.markdown("""
**Production Pipelines:**
* **Satisfaction Classifier:** `LightGBM (AUC ~0.72)`
* **Repeat Buyer Predictor:** `LightGBM (Top Decile Focus)`
* **Customer Clustering:** `K-Means (4 Personas)`

---
**Model Thresholds:**
* Friction Alert Cutoff: **$\ge 0.5397$**
* VIP Trigger Cutoff: **$\ge 0.0500$**
* Event Routing: **Discord Async Webhooks**
""")

# ==========================================
# TOP HERO SECTION
# ==========================================
st.title("Olist Marketplace Intelligence Platform")
st.markdown("<p style='font-size:1.15rem; color:#8b949e; margin-bottom:20px;'>Pre-dispatch customer friction mitigation, CRM repeat-order prediction, and operational analytics.</p>", unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown("""<div class='kpi-container'>
        <div class='kpi-title'>Orders Modeled</div>
        <div class='kpi-value'>99,441</div>
        <div class='kpi-sub'>Brazilian Cohort</div>
    </div>""", unsafe_allow_html=True)
with k2:
    st.markdown("""<div class='kpi-container'>
        <div class='kpi-title'>Baseline Friction</div>
        <div class='kpi-value'>14.8%</div>
        <div class='kpi-sub'>Review &le; 2 Stars</div>
    </div>""", unsafe_allow_html=True)
with k3:
    st.markdown("""<div class='kpi-container'>
        <div class='kpi-title'>Natural Repeat Rate</div>
        <div class='kpi-value'>3.1%</div>
        <div class='kpi-sub'>Marketplace Baseline</div>
    </div>""", unsafe_allow_html=True)
with k4:
    st.markdown("""<div class='kpi-container'>
        <div class='kpi-title'>Calibrated Threshold</div>
        <div class='kpi-value'>0.5397</div>
        <div class='kpi-sub'>Optimal Cost Boundary</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Platform Analytics", 
    "👥 Customer RFM Segments", 
    "⭐ VIP Repeat Predictor", 
    "🚨 Friction Risk Simulator", 
    "💰 Strategic ROI Engine"
])

# ------------------------------------------------------------------------------
# TAB 1: Platform Analytics
# ------------------------------------------------------------------------------
with tab1:
    st.header("Logistics Performance & SLA Latency")
    col1, col2 = st.columns(2)
    with col1:
        state_data = pd.DataFrame({
            "State": ["SP", "PR", "MG", "RJ", "SC", "RS", "DF", "BA", "GO", "PE", "CE", "PA", "MA", "AM", "RR"],
            "Avg_Delivery_Days": [8.3, 11.5, 11.6, 14.8, 14.5, 14.9, 12.5, 18.8, 15.2, 19.1, 20.8, 23.4, 21.2, 26.0, 29.3],
            "Friction_Rate": [0.08, 0.11, 0.12, 0.17, 0.13, 0.14, 0.13, 0.21, 0.15, 0.23, 0.24, 0.28, 0.27, 0.31, 0.38]
        })
        fig_geo = px.bar(
            state_data, x="State", y="Avg_Delivery_Days", color="Friction_Rate",
            color_continuous_scale="Reds", title="<b>Average Delivery Window & Friction Rate by State</b>",
            labels={"Avg_Delivery_Days": "Transit Days", "Friction_Rate": "Friction Ratio"},
            template="plotly_dark"
        )
        fig_geo.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#161b22", font_color="#e6edf3")
        st.plotly_chart(fig_geo, use_container_width=True)
        
    with col2:
        cohort_months = [f"Month {i}" for i in range(1, 13)]
        retention_rate = [100.0, 3.8, 2.9, 2.4, 2.1, 1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3]
        fig_cohort = px.line(
            x=cohort_months, y=retention_rate, markers=True,
            title="<b>Customer Retention Decay (Overall Cohort)</b>",
            labels={"x": "Cohort Window", "y": "Retention (%)"},
            template="plotly_dark"
        )
        fig_cohort.update_traces(line_color="#38bdf8", line_width=3, marker=dict(size=8, color="#818cf8"))
        fig_cohort.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#161b22", font_color="#e6edf3")
        st.plotly_chart(fig_cohort, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 2: Customer Segmentation
# ------------------------------------------------------------------------------
with tab2:
    st.header("RFM Behavioral Segments (K-Means Clustering)")
    col1, col2 = st.columns([1.1, 1.9])
    with col1:
        segment_counts = pd.DataFrame({
            "Segment": ["Recent Active", "Loyal High-Value", "Dormant Value", "Lost Transactional"],
            "Customers": [42500, 8900, 24100, 23900],
            "Avg_Spend": [112.5, 345.8, 189.2, 78.4]
        })
        fig_pie = px.pie(
            segment_counts, values="Customers", names="Segment", hole=0.55,
            title="<b>Customer Base Composition</b>",
            color_discrete_sequence=["#38bdf8", "#10b981", "#f59e0b", "#ef4444"],
            template="plotly_dark"
        )
        fig_pie.update_layout(paper_bgcolor="#161b22", font_color="#e6edf3")
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with col2:
        fig_scatter = px.scatter(
            segment_counts, x="Avg_Spend", y="Customers", size="Avg_Spend", color="Segment",
            color_discrete_sequence=["#38bdf8", "#10b981", "#f59e0b", "#ef4444"],
            title="<b>Monetary Contribution vs Population Size</b>",
            labels={"Avg_Spend": "Average Spend (R$)", "Customers": "Total Customers"},
            template="plotly_dark"
        )
        fig_scatter.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#161b22", font_color="#e6edf3")
        st.plotly_chart(fig_scatter, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 3: VIP Repeat Predictor
# ------------------------------------------------------------------------------
with tab3:
    st.header("⭐ VIP Repeat Buyer Predictor")
    st.markdown("Identifies high-propensity buyers and provides tailored retention campaigns based on order traits.")
    
    c1, c2 = st.columns([1.1, 1.3])
    with c1:
        st.subheader("First-Order Metrics")
        spend_m2 = st.number_input("First Order Spend (R$)", min_value=1.0, value=180.0, step=15.0, key="m2_spend")
        freight_m2 = st.number_input("Freight Amount (R$)", min_value=0.0, value=25.0, step=5.0, key="m2_freight")
        
        # Real-time Display
        m2_ratio = (freight_m2 / spend_m2) if spend_m2 > 0 else 0.0
        st.caption(f"Freight-to-Spend Ratio: **{m2_ratio:.2f}**")
        
        c_it, c_in = st.columns(2)
        with c_it:
            items_m2 = st.number_input("Items Count", min_value=1, max_value=15, value=1, key="m2_items")
        with c_in:
            inst_m2 = st.selectbox("Installments", [1, 2, 3, 4, 6, 8, 10, 12], index=2, key="m2_inst")
            
        c_pm, c_st = st.columns(2)
        with c_pm:
            pay_type_m2 = st.selectbox("Payment Type", ["credit_card", "boleto", "voucher", "debit_card"], key="m2_pay")
        with c_st:
            state_m2 = st.selectbox("Customer State", ["SP", "RJ", "MG", "RS", "PR", "BA", "SC", "DF", "GO", "PE"], key="m2_state")
            
        st.markdown("<br>", unsafe_allow_html=True)
        predict_repeat_btn = st.button("Evaluate VIP Likelihood 🔮", use_container_width=True)
        
    with c2:
        st.subheader("Model Decision & Dynamic CRM Playbook")
        if predict_repeat_btn:
            now = datetime.now()
            payload_m2 = {
                "first_order_spend": float(spend_m2),
                "first_order_freight": float(freight_m2),
                "first_order_freight_ratio": float(m2_ratio),
                "first_order_items_count": int(items_m2),
                "first_order_installments": int(inst_m2),
                "first_order_payment_type": pay_type_m2,
                "customer_state": state_m2,
                "purchase_month": now.month,
                "purchase_hour": now.hour,
                "purchase_dayofweek": now.weekday()
            }
            try:
                res = requests.post(REPEAT_PURCHASE_API_URL, json=payload_m2, timeout=4)
                if res.status_code == 200:
                    data = res.json()
                    prob = data.get("repeat_probability", 0.0)
                    is_vip = data.get("is_high_potential", False)
                    
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=prob * 100,
                        title={'text': "Repurchase Propensity (%)", 'font': {'size': 18, 'color': '#e6edf3'}},
                        gauge={
                            'axis': {'range': [0, 25], 'tickcolor': '#8b949e'},
                            'bar': {'color': "#10b981" if is_vip else "#38bdf8"},
                            'steps': [
                                {'range': [0, 5], 'color': "#21262d"},
                                {'range': [5, 25], 'color': "#1e3a2f"}
                            ],
                            'threshold': {
                                'line': {'color': "#f59e0b", 'width': 3},
                                'thickness': 0.75,
                                'value': 5.0
                            }
                        }
                    ))
                    fig_gauge.update_layout(paper_bgcolor="#161b22", height=240, margin=dict(l=20, r=20, t=30, b=10))
                    st.plotly_chart(fig_gauge, use_container_width=True)
                    
                    # Dynamic Input-Driven Recommendation Logic
                    if is_vip:
                        if spend_m2 >= 250:
                            vip_driver = "High Order Value (Whale Lead)"
                            action_plan = "Enroll into Executive Tier. Dispatch an unprompted R$ 30 thank-you voucher within 72 hours of successful delivery."
                        elif items_m2 > 1:
                            vip_driver = "Multi-Item Basket Intent"
                            action_plan = "Trigger category cross-sell campaign offering bundle discounts for their next checkout cycle."
                        else:
                            vip_driver = "Favorable Repeat Propensity Profile"
                            action_plan = "Dispatch targeted 10% repeat purchase coupon with priority customer support tagging."
                            
                        st.markdown(f"""
                        <div class='recommendation-card' style='border-left: 5px solid #10b981;'>
                            <div class='driver-tag' style='color:#10b981; border-color:#10b981;'>Key Retention Driver: {vip_driver}</div>
                            <h4 style='color:#f1f5f9; margin: 4px 0 8px 0;'>Recommended CRM Playbook:</h4>
                            <p style='color:#cbd5e1; font-size: 0.95rem; margin-bottom: 8px;'>{action_plan}</p>
                            <div class='event-badge'>⚡ Automated Lead Pushed to #crm-vip-leads</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class='recommendation-card' style='border-left: 5px solid #64748b;'>
                            <div class='driver-tag' style='color:#94a3b8; border-color:#64748b;'>Baseline Profile</div>
                            <h4 style='color:#f1f5f9; margin: 4px 0 8px 0;'>Recommended Playbook:</h4>
                            <p style='color:#94a3b8; font-size: 0.95rem; margin-bottom: 0;'>Propensity score is within the normal single-purchase baseline. Retain in standard automated quarterly newsletter loop.</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.error(f"Inference Gateway returned HTTP {res.status_code}")
            except Exception:
                st.error("Connection Failed: Verify FastAPI backend is active on port 8000.")
        else:
            st.info("Input first-order attributes to score customer retention potential.")

# ------------------------------------------------------------------------------
# TAB 4: Customer Satisfaction (Strictly Top Features from Table)
# ------------------------------------------------------------------------------
with tab4:
    st.header("🚨 Pre-Dispatch Friction Risk Simulator")
    st.markdown("Evaluates top LightGBM drivers and returns root-cause diagnosis with a targeted warehouse playbook.")
    
    col1, col2 = st.columns([1.2, 1.2])
    with col1:
        st.subheader("Top Feature Importance Inputs")
        
        # 1. Total Volume (Importance: 906) & Total Weight (Importance: 708)
        c_vol, c_wt = st.columns(2)
        with c_vol:
            volume = st.number_input("Parcel Volume (cm³)", min_value=100, value=12000, step=500, key="m1_vol", help="Rank 1: Importance 906")
        with c_wt:
            weight = st.number_input("Total Weight (grams)", min_value=50, value=2500, step=250, key="m1_wt", help="Rank 4: Importance 708")
            
        # 2. Estimated Transit Days (Importance: 782)
        transit = st.slider("Estimated Transit SLA (Days)", min_value=1, max_value=60, value=28, key="m1_transit", help="Rank 2: Importance 782")
        
        # 3. Order Price (Importance: 740) & Freight Value (Importance: 655)
        c_price, c_frt = st.columns(2)
        with c_price:
            price = st.number_input("Total Order Price (R$)", min_value=1.0, value=85.0, step=10.0, key="m1_price", help="Rank 3: Importance 740")
        with c_frt:
            frt = st.number_input("Total Freight Value (R$)", min_value=0.0, value=45.0, step=5.0, key="m1_frt", help="Rank 5: Importance 655")
            
        # 4. Auto-Computed Freight Ratio (Importance: 592)
        freight_ratio = (frt / price) if price > 0 else 0.0
        st.caption(f"Calculated `freight_to_price_ratio`: **{freight_ratio:.4f}** (Rank 6: Importance 592)")
        
        # 5. Payment Installments Max (Importance: 220) & Supplementary Details
        c_inst, c_cross = st.columns(2)
        with c_inst:
            installments = st.number_input("Payment Installments Max", min_value=1, max_value=24, value=2, key="m1_inst", help="Rank 9: Importance 220")
        with c_cross:
            cross_state = st.selectbox("Interstate Cross-Dock?", [1, 0], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)", key="m1_cross")
            
        st.markdown("<br>", unsafe_allow_html=True)
        predict_sat_btn = st.button("Evaluate Dispatch Risk 🔍", use_container_width=True)
        
    with col2:
        st.subheader("Model Risk Score & Root-Cause Playbook")
        if predict_sat_btn:
            now = datetime.now()
            # Clean payload: Top features + auto-computed ratios + current timestamp context
            payload_m1 = {
                "total_order_price": float(price),
                "total_freight_value": float(frt),
                "freight_to_price_ratio": float(freight_ratio),
                "total_items_count": 1,
                "total_weight_g": float(weight),
                "total_volume_cm3": float(volume),
                "estimated_transit_days": float(transit),
                "is_cross_state": int(cross_state),
                "payment_sequential_count": 1,
                "payment_installments_max": int(installments),
                "purchase_year": now.year,
                "purchase_month": now.month,
                "purchase_day": now.day,
                "purchase_dayofweek": now.weekday(),
                "purchase_hour": now.hour,
                "preferred_payment_type": "credit_card"
            }
            try:
                res = requests.post(SATISFACTION_API_URL, json=payload_m1, timeout=4)
                if res.status_code == 200:
                    data = res.json()
                    risk = data.get("dissatisfaction_risk", 0.0)
                    alert = data.get("alert_triggered", False)
                    
                    fig_risk = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=risk * 100,
                        title={'text': "Dissatisfaction Risk (%)", 'font': {'size': 18, 'color': '#e6edf3'}},
                        gauge={
                            'axis': {'range': [0, 100], 'tickcolor': '#8b949e'},
                            'bar': {'color': "#ef4444" if alert else "#10b981"},
                            'steps': [
                                {'range': [0, 53.97], 'color': "#1a2e22"},
                                {'range': [53.97, 100], 'color': "#381a1a"}
                            ],
                            'threshold': {
                                'line': {'color': "#ef4444", 'width': 3},
                                'thickness': 0.75,
                                'value': 53.97
                            }
                        }
                    ))
                    fig_risk.update_layout(paper_bgcolor="#161b22", height=240, margin=dict(l=20, r=20, t=30, b=10))
                    st.plotly_chart(fig_risk, use_container_width=True)
                    
                    # Dynamic Root-Cause Diagnosis Logic
                    if alert:
                        drivers = []
                        if freight_ratio > 0.40:
                            drivers.append("High Freight Friction (>40% of item value)")
                        if transit >= 25:
                            drivers.append(f"Extended SLA Window ({transit} Days)")
                        if volume >= 15000 or weight >= 3000:
                            drivers.append("Bulky Packaging Overhead")
                            
                        primary_driver = " & ".join(drivers) if drivers else "Compound Logistics Latency"
                        
                        # Feature-specific operational playbook
                        if transit >= 25 and freight_ratio > 0.40:
                            warehouse_action = "Reroute package to Tier-1 Express Carrier and attach proactive SMS tracking dispatch update."
                        elif volume >= 15000:
                            warehouse_action = "Execute physical warehouse repackaging audit to reinforce protective cushioning before handover."
                        else:
                            warehouse_action = "Flag priority fulfillment queue and verify carrier dispatch timestamp within 4 hours."
                            
                        st.markdown(f"""
                        <div class='recommendation-card' style='border-left: 5px solid #ef4444;'>
                            <div class='driver-tag' style='color:#ef4444; border-color:#ef4444;'>Identified Root-Cause: {primary_driver}</div>
                            <h4 style='color:#f1f5f9; margin: 4px 0 8px 0;'>Mandated Warehouse Protocol:</h4>
                            <p style='color:#cbd5e1; font-size: 0.95rem; margin-bottom: 8px;'>{warehouse_action}</p>
                            <div class='event-badge'>⚡ Dispatch Alert Pushed to #warehouse-alerts</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class='recommendation-card' style='border-left: 5px solid #10b981;'>
                            <div class='driver-tag' style='color:#10b981; border-color:#10b981;'>Logistics Metrics: Optimal</div>
                            <h4 style='color:#f1f5f9; margin: 4px 0 8px 0;'>Standard Fulfillment Clearance:</h4>
                            <p style='color:#10b981; font-size: 0.95rem; margin-bottom: 0;'>Risk score is below decision boundary (0.5397). Route through automated standard packing line.</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.error(f"Inference Gateway returned HTTP {res.status_code}")
            except Exception:
                st.error("Connection Failed: Verify FastAPI backend is active on port 8000.")
        else:
            st.info("Set order attributes to evaluate against the trained LightGBM pipeline.")

# ------------------------------------------------------------------------------
# TAB 5: Strategic Business ROI Engine
# ------------------------------------------------------------------------------
with tab5:
    st.header("💰 Executive ROI & Cost-Benefit Simulator")
    st.markdown("Simulate net savings from automated pre-dispatch interventions.")
    
    col1, col2 = st.columns([1, 1.4])
    with col1:
        st.subheader("Financial Assumptions")
        order_volume = st.slider("Annual Order Ingestion", 50000, 500000, 100000, step=10000)
        churn_cost = st.number_input("Support & Churn Cost / Friction Case (R$)", min_value=10.0, value=85.0)
        action_cost = st.number_input("Express Upgrade Cost / Parcel (R$)", min_value=1.0, value=12.0)
        success_rate = st.slider("Intervention Remediation Rate (%)", 10.0, 100.0, 65.0, step=5.0) / 100
        
    with col2:
        st.subheader("Projected Financial Impact")
        base_friction = 0.148
        model_precision, model_recall = 0.46, 0.62
        
        actual_dissatisfied = order_volume * base_friction
        baseline_cost = actual_dissatisfied * churn_cost
        
        true_positives = actual_dissatisfied * model_recall
        alerts_triggered = true_positives / model_precision
        intervention_spend = alerts_triggered * action_cost
        
        prevented_cases = true_positives * success_rate
        new_total_cost = intervention_spend + ((actual_dissatisfied - prevented_cases) * churn_cost)
        
        net_savings = baseline_cost - new_total_cost
        roi = (net_savings / intervention_spend) * 100 if intervention_spend > 0 else 0
        
        waterfall_df = pd.DataFrame({
            "Stage": ["Baseline Friction", "Intervention Spend", "Remediated Savings", "Net Savings"],
            "Amount": [baseline_cost, -intervention_spend, baseline_cost - (new_total_cost - intervention_spend), net_savings]
        })
        
        fig_waterfall = px.bar(
            waterfall_df, x="Stage", y="Amount", color="Amount", color_continuous_scale="Tealgrn",
            title="<b>Financial Breakdown (BRL R$)</b>", template="plotly_dark"
        )
        fig_waterfall.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#161b22", font_color="#e6edf3")
        st.plotly_chart(fig_waterfall, use_container_width=True)
        
        st.markdown(f"""
        <div style='background-color:#161b22; border:1px solid #30363d; border-radius:12px; padding:20px;'>
            <h2 style='color:#10b981; margin:0;'>Net Projected Savings: R$ {net_savings:,.2f}</h2>
            <p style='color:#e6edf3; font-size:1.15rem; margin-top:6px;'>
                Operational Return on Investment: <b style='color:#38bdf8;'>{roi:,.1f}%</b>
            </p>
            <ul style='color:#8b949e; font-size:0.95rem; line-height:1.7; margin-bottom:0;'>
                <li>Annual Targeted Package Upgrades: <b>{int(alerts_triggered):,}</b></li>
                <li>Dissatisfaction Cases Successfully Averted: <b>{int(prevented_cases):,}</b></li>
                <li>Estimated Total Intervention Cost: <b>R$ {intervention_spend:,.2f}</b></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)