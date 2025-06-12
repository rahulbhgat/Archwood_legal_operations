# services/analytics.py

import pandas as pd

def compute_dashboard_metrics(attorneys, clients, matters, leaves):
    metrics = {}

    # KPI Cards
    metrics['total_attorneys'] = len(attorneys)
    metrics['active_clients'] = len(clients[clients['status'].str.lower() == 'active'])
    metrics['open_matters'] = len(matters[matters['status'].str.lower() == 'active'])
    metrics['pending_leaves'] = len(leaves[leaves['approval_status'].str.lower() == 'pending'])
    metrics['billing_anomalies'] = pd.DataFrame()

    # Recent matters
    matters['open_date'] = pd.to_datetime(matters['open_date'], errors='coerce')
    metrics['recent_matters'] = matters.sort_values(by='open_date', ascending=False).head(5)

    # Upcoming leaves
    leaves['start_date'] = pd.to_datetime(leaves['start_date'], errors='coerce')
    metrics['upcoming_leaves'] = leaves[leaves['start_date'] >= pd.Timestamp.today()].sort_values(by='start_date').head(5)

    # Top clients by matter count
    top_clients = matters['client_id'].value_counts().head(5).reset_index()
    top_clients.columns = ['client_id', 'matter_count']
    metrics['top_clients'] = pd.merge(top_clients, clients, on='client_id', how='left')

    # AI: Workload Anomaly Detection
    metrics['attorney_workload_anomalies'] = detect_attorney_workload_anomalies(matters, attorneys)

    return metrics


def detect_attorney_workload_anomalies(matters, attorneys):
    # Calculate average complexity of active matters per attorney
    active_matters = matters[matters['status'].str.lower() == 'active']
    workload = active_matters.groupby('attorney_id')['complexity_score'].mean().reset_index()
    workload.columns = ['attorney_id', 'avg_complexity']

    # Calculate threshold: consider top 10% as anomalies
    threshold = workload['avg_complexity'].quantile(0.9)
    anomalies = workload[workload['avg_complexity'] > threshold]

    # Merge with attorney names for display
    anomalies = pd.merge(anomalies, attorneys, on='attorney_id', how='left')

    return anomalies
