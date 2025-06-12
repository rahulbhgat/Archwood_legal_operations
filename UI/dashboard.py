import streamlit as st
import pandas as pd
from services.data_loader import load_all_data
from services.analytics import compute_dashboard_metrics
import altair as alt

def display_dashboard():
    # Load data from CSV files
    attorneys, clients, matters, leaves = load_all_data()

    # Compute all metrics
    metrics = compute_dashboard_metrics(attorneys, clients, matters, leaves)

    # Dashboard Title
    st.markdown("## 📊 Legal ERP Dashboard")
    st.markdown("Welcome to the central dashboard. Below are key metrics and tables.")

    # KPI Cards
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("👩‍⚖️ Total Attorneys", metrics['total_attorneys'])
    col2.metric("🧑‍💼 Active Clients", metrics['active_clients'])
    col3.metric("📂 Open Matters", metrics['open_matters'])
    col4.metric("🕒 Pending Leaves", metrics['pending_leaves'])

    st.markdown("---")

    # Section: Recent Matters
    st.markdown("### 🗂️ Recent Matters")
    st.dataframe(metrics['recent_matters'].style.set_properties(**{'text-align': 'left'}), use_container_width=True)

    # Section: Upcoming Leaves
    st.markdown("### 🏖️ Upcoming Leaves")
    st.dataframe(metrics['upcoming_leaves'], use_container_width=True)

    # Section: Top Clients
    st.markdown("### 🤝 Top Clients by Matter Count")
    st.dataframe(metrics['top_clients'], use_container_width=True)

    # Section: Attorney Workload Anomalies
    st.markdown("### ⚠️ Attorney Workload Anomalies")
    if not metrics['attorney_workload_anomalies'].empty:
        st.dataframe(metrics['attorney_workload_anomalies'], use_container_width=True)
    else:
        st.success("✅ No workload anomalies detected.")

    # Section: Billing Rate Anomalies
    st.markdown("### 💸 Billing Rate Anomalies")
    if not metrics['billing_anomalies'].empty:
        st.dataframe(metrics['billing_anomalies'], use_container_width=True)
    else:
        st.success("✅ No billing anomalies detected.")
