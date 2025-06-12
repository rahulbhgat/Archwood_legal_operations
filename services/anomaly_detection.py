import pandas as pd
import os

# Load CSVs (update the paths as needed)
BASE_PATH = "data"
matter_df = pd.read_csv(os.path.join(BASE_PATH, "matter_data.csv"))
leave_df = pd.read_csv(os.path.join(BASE_PATH, "leave_time_data.csv"))
client_df = pd.read_csv(os.path.join(BASE_PATH, "client_data.csv"))
attorney_df = pd.read_csv(os.path.join(BASE_PATH, "attorney_data.csv"))

# ------------------ Anomaly Rules ------------------
def detect_matter_anomalies(matter_df):
    anomalies = []

    for idx, row in matter_df.iterrows():
        # Rule 1: Estimated vs Actual Close Date delay
        if pd.notna(row['estimated_close_date']) and pd.notna(row['actual_close_date']):
            est = pd.to_datetime(row['estimated_close_date'])
            act = pd.to_datetime(row['actual_close_date'])
            if act > est:
                anomalies.append({
                    "type": "Matter Delay",
                    "id": row['matter_id'],
                    "description": f"Matter {row['matter_id']} closed later than estimated by {(act - est).days} days."
                })

        # Rule 2: Matters with no actual close date and open too long
        if pd.isna(row['actual_close_date']) and pd.notna(row['open_date']):
            open_date = pd.to_datetime(row['open_date'])
            if (pd.Timestamp.now() - open_date).days > 365:
                anomalies.append({
                    "type": "Stale Open Matter",
                    "id": row['matter_id'],
                    "description": f"Matter {row['matter_id']} has been open for more than 1 year without closure."
                })

    return anomalies

def detect_leave_anomalies(leave_df):
    anomalies = []
    leave_df['start_date'] = pd.to_datetime(leave_df['start_date'])
    leave_df['end_date'] = pd.to_datetime(leave_df['end_date'])

    # Rule: Leave duration more than 10 days
    for idx, row in leave_df.iterrows():
        duration = (row['end_date'] - row['start_date']).days
        if duration > 10:
            anomalies.append({
                "type": "Extended Leave",
                "id": row['leave_id'],
                "description": f"Attorney {row['attorney_id']} took leave for {duration} days."
            })

    return anomalies

def detect_client_anomalies(client_df):
    anomalies = []
    # Rule: Clients marked as active but have no open matters (joined with matter_df)
    active_clients = client_df[client_df['ststus'].str.lower() == 'active']
    open_matters = matter_df[matter_df['status'].str.lower() == 'active']

    for idx, row in active_clients.iterrows():
        if row['client_id'] not in open_matters['client_id'].values:
            anomalies.append({
                "type": "Inactive Client",
                "id": row['client_id'],
                "description": f"Client {row['client_id']} is marked active but has no open matters."
            })

    return anomalies

# ------------------ Aggregate ------------------
def detect_all_anomalies():
    anomalies = []
    anomalies.extend(detect_matter_anomalies(matter_df))
    anomalies.extend(detect_leave_anomalies(leave_df))
    anomalies.extend(detect_client_anomalies(client_df))
    return anomalies
