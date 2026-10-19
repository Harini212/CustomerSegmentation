import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler  # Import StandardScaler

# --- File Paths ---
MODEL_PATH = 'kmeans_model.pkl'
SCALER_PATH = 'scaler.pkl'
ENCODING_PATH = 'encoding.pkl'

# Load the model
with open(MODEL_PATH, 'rb') as file:
    kmeans_loaded = pickle.load(file)

# Load the scaler
with open(SCALER_PATH, 'rb') as file:
    scaler_loaded = pickle.load(file)

# Load the encoding
with open(ENCODING_PATH, 'rb') as file:
    cluster_labels_encoding_loaded = pickle.load(file)


# Example usage: Predict cluster for new data
new_data = pd.DataFrame({'Normalized_Recency': [-0.2], 'Normalized_Frequency': [-0.6], 'Normalized_MonetaryValue': [-0.9]}) # Replace with your new data
new_data_scaled = scaler_loaded.transform(new_data)  # Scale the new data using the loaded scaler
cluster_prediction = kmeans_loaded.predict(new_data_scaled)[0]  # Predict the cluster
segment_label = cluster_labels_encoding_loaded.get(cluster_prediction, "Unknown") # Get label or "Unknown"
print(f"The predicted cluster is: {cluster_prediction}, Segment Label: {segment_label}")
