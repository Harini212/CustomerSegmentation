import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score
import scipy.stats as stats
import pickle  # Import pickle for saving the model components

# --- File Paths for Saving ---
MODEL_PATH = 'hierarchical_model.pkl'
SCALER_PATH = 'scaler.pkl'
ENCODING_PATH = 'encoding.pkl'  # Path to save the cluster labels encoding

# Load the data
data = pd.read_csv('preprocessed.csv')  # Replace 'preprocessed.csv' with your file
# Make sure your CSV includes 'CustomerID' for aggregation

# Select features for clustering
features = ['Normalized_Recency', 'Normalized_Frequency', 'Normalized_MonetaryValue']  # Replace with your features

# --- Reduce Dataset Size by Sampling ---
sample_size = 10000  # Adjust this value based on your memory constraints
if len(data) > sample_size:
    data_sample = data.sample(n=sample_size, random_state=0)  # Use random_state for reproducibility
    X = data_sample[features]
else:
    X = data[features]
    data_sample = data.copy()  # Create data_sample when not sampling


# Apply scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Hierarchical Clustering ---
linkage_method = 'ward'
linked = linkage(X_scaled, method=linkage_method)

# Determine optimal k using silhouette analysis (as before)
silhouette_scores = []
range_n_clusters = range(2, min(11, len(X_scaled) - 1)) #Ensure k is less than the number of samples
if len(X_scaled) > 1: #Ensure you have more than one sample
    for n_clusters in range_n_clusters:
        cluster_labels = fcluster(linked, n_clusters, criterion='maxclust')
        silhouette_avg = silhouette_score(X_scaled, cluster_labels)
        silhouette_scores.append(silhouette_avg)

    # Print silhouette scores
    print("Silhouette Scores for Different Numbers of Clusters:")
    for n_clusters, score in zip(range_n_clusters, silhouette_scores):
        print(f"Clusters: {n_clusters}, Silhouette Score: {score}")

    # Get optimal k from the plot or by finding the maximum silhouette score
    optimal_k = int(input("Enter the optimal K value based on the silhouette scores: "))

    # Apply fcluster to assign data points to clusters
    data_sample['Hierarchical_Cluster'] = fcluster(linked, optimal_k, criterion='maxclust') #Assign clusters to the sample

    print(data_sample.groupby('Hierarchical_Cluster')[features].mean()) #Print cluster means

    # --- Segment Analysis and Interpretation ---

    # 1. Descriptive Statistics for Key Features by Cluster
    cluster_summary = data_sample.groupby('Hierarchical_Cluster')[features].agg(['mean', 'median', 'std', 'min', 'max'])
    print("\nDescriptive Statistics by Cluster:\n", cluster_summary)

    # 2. Add other relevant variables (example with a hypothetical 'Country' column)
    # Assuming your data has a column named 'Country'
    if 'Country' in data_sample.columns:
        Country_counts = data_sample.groupby(['Hierarchical_Cluster', 'Country']).size().unstack(fill_value=0)
        print("\nCountry Distribution by Cluster:\n", Country_counts)

        # Chi-square test for independence (Cluster vs. Country)
        chi2, p, dof, expected = stats.chi2_contingency(Country_counts)
        print(f"\nChi-square test for independence (Cluster vs. Country):")
        print(f"Chi2 statistic: {chi2}, p-value: {p}")
        if p < 0.05:  # Or your chosen significance level
            print("The distribution of Countries is significantly different across clusters.")
        else:
            print("There is no significant association between cluster and Country.")
    else:
        print("No Country data")

    #3. Customer Count by Cluster
    customer_counts = data_sample.groupby('Hierarchical_Cluster')['CustomerID'].count()
    print("\nCustomer Count by Cluster:\n", customer_counts)

    # 4. Labeling the Segments (Requires Manual Interpretation)
    # Based on the descriptive statistics and other analysis, label each cluster.
    # This part requires your domain knowledge and understanding of the data.
    cluster_labels = {
        1: "VIP Customers",  # Replace with your actual labels
        2: "Lost Customers",
        3: "Potential Customers",
        # Add more labels as needed based on your optimal_k
    }
    data_sample['Segment_Label'] = data_sample['Hierarchical_Cluster'].map(cluster_labels)

     # Create a reverse mapping from cluster labels to cluster names
    cluster_labels_encoding = {cluster: label for cluster, label in cluster_labels.items()}

    print("\nSegment Labels:\n", data_sample[['Hierarchical_Cluster', 'Segment_Label']].head())

    # Display the first few rows with the cluster assignments and segment labels
    print("\nData with Cluster and Segment Labels:\n", data_sample.head())

    # 5. Example persona generation

    def generate_persona(cluster_num, cluster_data):
        """Generates a simple persona description based on cluster data."""
        cluster_desc = cluster_data.describe()
        avg_recency = cluster_desc.loc['mean', 'Normalized_Recency']
        avg_frequency = cluster_desc.loc['mean', 'Normalized_Frequency']
        avg_monetary = cluster_desc.loc['mean', 'Normalized_MonetaryValue']

        # Simple conditions to define persona traits (improve as needed)
        if avg_recency < 0.3:
            recency_desc = "Recently active"
        else:
            recency_desc = "Not recently active"

        if avg_frequency > 0.5:
            frequency_desc = "Frequent buyer"
        else:
            frequency_desc = "Infrequent buyer"

        if avg_monetary > 0.5:
            monetary_desc = "High spender"
        else:
            monetary_desc = "Low spender"

        persona = (f"Cluster {cluster_num} Persona: {recency_desc}, {frequency_desc}, {monetary_desc}.  "
                   f"Recency: {avg_recency:.2f}, Frequency: {avg_frequency:.2f}, Monetary Value: {avg_monetary:.2f}")

        return persona

    # Generate personas for each cluster
    for cluster in data_sample['Hierarchical_Cluster'].unique():
        cluster_data = data_sample[data_sample['Hierarchical_Cluster'] == cluster][features]
        persona = generate_persona(cluster, cluster_data)
        print(f"\n{persona}")

    # --- Saving the Model Components ---
    with open(MODEL_PATH, 'wb') as file:
        pickle.dump(linked, file)  # Save the linkage matrix

    with open(SCALER_PATH, 'wb') as file:
        pickle.dump(scaler, file)  # Save the scaler

    with open(ENCODING_PATH, 'wb') as file: #Store the clusters labels encoding
        pickle.dump(cluster_labels_encoding, file)

else:
    print ("Not enough samples in data")
