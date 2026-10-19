import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
import pickle

# Load data and sample for speed
data = pd.read_csv('preprocessed.csv')
sample_size = 5000
if len(data) > sample_size:
    data_sample = data.sample(n=sample_size, random_state=0)
else:
    data_sample = data.copy()

features = ['Normalized_Recency', 'Normalized_Frequency', 'Normalized_MonetaryValue']
X = data_sample[features].values

# Load scaler and scale
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
X_scaled = scaler.transform(X)

def evaluate_clustering(labels, X_scaled):
    print(f"Davies-Bouldin Index: {davies_bouldin_score(X_scaled, labels):.4f}")
    print(f"Calinski-Harabasz Index: {calinski_harabasz_score(X_scaled, labels):.4f}")

# Load KMeans model & predict clusters
with open('kmeans_model.pkl', 'rb') as f:
    kmeans_model = pickle.load(f)
kmeans_labels = kmeans_model.predict(X_scaled)
print("Evaluating K-Means Clustering:")
evaluate_clustering(kmeans_labels, X_scaled)

print("\n---------------------------------\n")

# Load hierarchical clustering labels (just labels, no model)
with open('hierarchical_model.pkl', 'rb') as f:
    hierarchical_labels = pickle.load(f)  # Assuming labels are saved

print("Evaluating Hierarchical Clustering:")
evaluate_clustering(hierarchical_labels, X_scaled)
