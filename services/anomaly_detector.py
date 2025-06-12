# services/anomaly_detector.py

import pandas as pd
from sklearn.ensemble import IsolationForest

def detect_attorney_workload_anomalies(matters_df, attorneys_df):
    """
    Detect attorneys handling too many matters using Isolation Forest.
    """
    # Count how many matters each attorney is handling
    workload = matters_df['Attorney'].value_counts().reset_index()
    workload.columns = ['Attorney', 'MatterCount']

    # Apply Isolation Forest
    model = IsolationForest(contamination=0.1, random_state=42)
    workload['anomaly'] = model.fit_predict(workload[['MatterCount']])
    anomalies = workload[workload['anomaly'] == -1]

    return anomalies[['Attorney', 'MatterCount']]

def detect_unusual_client_activity(matters_df, clients_df):
    """
    Detect clients with suspiciously low or high number of matters.
    """
    activity = matters_df['Client'].value_counts().reset_index()
    activity.columns = ['Client', 'MatterCount']

    model = IsolationForest(contamination=0.1, random_state=42)
    activity['anomaly'] = model.fit_predict(activity[['MatterCount']])
    anomalies = activity[activity['anomaly'] == -1]

    return anomalies[['Client', 'MatterCount']]
