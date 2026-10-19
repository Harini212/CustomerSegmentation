import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans  # Import KMeans
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # Import for 3D plotting

# --- File Paths (Ensure these are correct!) ---
DATA_PATH = 'preprocessed.csv'
MODEL_PATH = 'kmeans_model.pkl'
SCALER_PATH = 'scaler.pkl'
ENCODING_PATH = 'encoding.pkl'

# --- Load Data and Trained Components ---

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

def load_model(path):
    with open(path, 'rb') as file:
        return pickle.load(file)

data = load_data(DATA_PATH)
scaler = load_model(SCALER_PATH)
kmeans_model = load_model(MODEL_PATH)  # Load KMeans model
cluster_labels_encoding = load_model(ENCODING_PATH)

# Select features (must match training)
features = ['Normalized_Recency', 'Normalized_Frequency', 'Normalized_MonetaryValue']

# --- Sample the data (same sampling as training) ---
sample_size = 10000  # Should match your training script
if len(data) > sample_size:
    data_sample = data.sample(n=sample_size, random_state=0)
    X = data_sample[features]
else:
    X = data[features]
    data_sample = data.copy()

# Apply scaling (same scaler as training)
X_scaled = scaler.transform(X)

# Assign cluster labels to sampled data (using the pre-trained KMeans model)
data_sample['KMeans_Cluster'] = kmeans_model.predict(X_scaled)  # Use KMeans to predict

# --- Visualizations (Using Sampled Data) ---

st.title("Customer Segmentation Analysis")

st.header("Cluster Visualizations")

# 1. Scatter Plot (2D)
st.subheader("2D Scatter Plot")
feature1 = st.selectbox("Select Feature 1 for Scatter Plot:", features, index=0)
feature2 = st.selectbox("Select Feature 2 for Scatter Plot:", features, index=1)

fig_scatter, ax_scatter = plt.subplots(figsize=(8, 6))
sns.scatterplot(x=feature1, y=feature2, hue='KMeans_Cluster', data=data_sample, palette='viridis', ax=ax_scatter)
ax_scatter.set_title(f'{feature1} vs {feature2} (Clusters)')
st.pyplot(fig_scatter)

# 2. 3D Scatter Plot
st.subheader("3D Scatter Plot")

fig_3d = plt.figure(figsize=(10, 8))
ax_3d = fig_3d.add_subplot(111, projection='3d')

x = data_sample['Normalized_Recency']
y = data_sample['Normalized_Frequency']
z = data_sample['Normalized_MonetaryValue']

scatter = ax_3d.scatter(x, y, z, c=data_sample['KMeans_Cluster'], cmap='viridis', marker='o', s=50)

ax_3d.set_xlabel('Normalized_Recency')
ax_3d.set_ylabel('Normalized_Frequency')
ax_3d.set_zlabel('Normalized_MonetaryValue')
ax_3d.set_title('3D Scatter Plot of KMeans Clusters')

# Add a legend
legend = ax_3d.legend(*scatter.legend_elements(), loc="upper right", title="Clusters")
ax_3d.add_artist(legend)

st.pyplot(fig_3d)

# 3. Box Plots
st.subheader("Box Plots by Cluster")
selected_feature = st.selectbox("Select Feature for Box Plot:", features)

fig_box, ax_box = plt.subplots(figsize=(8, 6))
sns.boxplot(x='KMeans_Cluster', y=selected_feature, data=data_sample, palette='viridis', ax=ax_box)
ax_box.set_title(f'{selected_feature} by Cluster')
st.pyplot(fig_box)

# --- Customer Segmentation and Recommendations ---

st.header("Customer Segmentation and Recommendations")

customer_id = st.text_input("Enter CustomerID:", "")

def predict_segment(customer_id, data, scaler, kmeans_model, features, encoding):
    """Predicts the customer segment for a given CustomerID using KMeans."""

    if int(customer_id) not in data['CustomerID'].values:
        return "Customer ID not found."

    customer_data = data[data['CustomerID'] == int(customer_id)][features]
    st.write("Customer Data:", customer_data)  # Add this line

    # Scale the features
    customer_scaled = scaler.transform(customer_data)
    st.write("Scaled Customer Data:", customer_scaled) # Add this line

    # Predict the cluster using KMeans
    cluster_assignment = kmeans_model.predict(customer_scaled)[0] # Predict using KMeans
    st.write("Cluster Assignment:", cluster_assignment) # Add this line

    # Get the segment label
    segment_label = encoding.get(cluster_assignment, "Unknown")
    st.write("Segment Label", segment_label)
    #st.write("Cluster Labels Encoding:", cluster_labels_encoding)  # Corrected variable name

    return segment_label


def get_recommendations(segment_label):
    """Provides recommendations based on the predicted segment."""

    recommendations = {
        "Potential Customers": [
            "Offer personalized discounts to encourage repeat purchases.",
            "Recommend products that align with their past buying behavior.",
            "Highlight the value and affordability of your offerings."
        ],
        "Lost Customers": [
            "Send a personalized re-engagement email with a special offer.",
            "Offer a discount on their next purchase to win them back.",
            "Ask for feedback to understand why they left and how you can improve."
        ],
        "VIP Customers": [
            "Implement a loyalty program with exclusive rewards and benefits.",
            "Provide personalized product recommendations and early access to new products.",
            "Offer VIP customer service and support."
        ]
    }

    return recommendations.get(segment_label, ["No specific recommendations available for this segment."])

if customer_id:
    try:
        customer_id = int(customer_id)  # Ensure it's an integer
        segment_label = predict_segment(customer_id, data, scaler, kmeans_model, features, cluster_labels_encoding)

        st.subheader("Prediction:")
        st.write(f"CustomerID {customer_id} belongs to the **{segment_label}** segment.")

        st.subheader("Recommendations:")
        recs = get_recommendations(segment_label)
        for rec in recs:
            st.write(f"- {rec}")

    except ValueError:
        st.error("Invalid CustomerID. Please enter a valid integer.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
