# File: services/data_loader.py

import pandas as pd
import os

def load_all_data():
    base_path = os.path.join(os.getcwd(),'Data')

    attorneys = pd.read_csv(os.path.join(base_path, 'attorney_data.csv'))
    clients = pd.read_csv(os.path.join(base_path, 'client_data.csv'))
    matters = pd.read_csv(os.path.join(base_path, 'matter_data.csv'))
    leaves = pd.read_csv(os.path.join(base_path, 'leave_time_data.csv'))

    return attorneys, clients, matters, leaves
