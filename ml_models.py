from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
import numpy as np

def train_risk_model(X, y):
    model = RandomForestClassifier()
    model.fit(X, y)
    return model

def predict_risk(model, X_new):
    return model.predict(X_new)

def train_segmentation(X, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X)
    return kmeans

def predict_segment(kmeans, X_new):
    return kmeans.predict(X_new)

# Net flow prediction (dummy, replace with ARIMA/LSTM/Prophet as needed)
def predict_net_flow(std_df):
    # Simple net flow: sum of credits - sum of debits
    net_flow = std_df[std_df['Type'] == 'Credit']['Amount'].sum() - abs(std_df[std_df['Type'] == 'Debit']['Amount'].sum())
    return net_flow 