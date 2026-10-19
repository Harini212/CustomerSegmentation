import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # Import for 3D plotting

data = pd.read_csv('preprocessed.csv')  # Replace 'preprocessed.csv' with your file
 
 # Select features for clustering
features = ['Normalized_Recency', 'Normalized_Frequency', 'Normalized_MonetaryValue']  # Replace with your features
X = data[features]

# Apply scaling AGAIN
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 1. Elbow Method for Optimal k
def plot_elbow_method(X, max_clusters=10):
 wcss = []  # Within-Cluster Sum of Squares
 for i in range(1, max_clusters + 1):
  kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
  kmeans.fit(X)
  wcss.append(kmeans.inertia_)

 plt.plot(range(1, max_clusters + 1), wcss, marker='o')
 plt.title('Elbow Method for Optimal k')
 plt.xlabel('Number of Clusters (k)')
 plt.ylabel('WCSS')
 plt.show()
'''
# 2. Silhouette Score for Optimal k
def plot_silhouette_analysis(X, max_clusters=10):
 silhouette_scores = []
 for n_cluster in range(2, max_clusters + 1):
  kmeans = KMeans(n_clusters=n_cluster, init='k-means++', max_iter=300, n_init=10, random_state=0)
  cluster_labels = kmeans.fit_predict(X)
  silhouette_avg = silhouette_score(X, cluster_labels)
  silhouette_scores.append(silhouette_avg)
 

 plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
 plt.title('Silhouette Analysis for Optimal k')
 plt.xlabel('Number of Clusters (k)')
 plt.ylabel('Silhouette Score')
 plt.show()
 '''

plot_elbow_method(X_scaled)
#plot_silhouette_analysis(X_scaled)

# 3. Apply K-Means with the optimal number of clusters
optimal_k = int(input('Enter the Optimal K value: '))  
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=300, n_init=10, random_state=0)
data['KMeans_Cluster'] = kmeans.fit_predict(X_scaled)

print(data.groupby('KMeans_Cluster')[features].mean())
print(data.head())

# --- Visualizations ---

# 1. Scatter Plot (2D) - Choose two features to visualize
feature1 = 'Normalized_Recency'
feature2 = 'Normalized_Frequency'
 

plt.figure(figsize=(8, 6))
sns.scatterplot(x=feature1, y=feature2, hue='KMeans_Cluster', data=data, palette='viridis')
plt.title(f'{feature1} vs {feature2} (Clusters)')
plt.xlabel(feature1)
plt.ylabel(feature2)
plt.show()

# 2. 3D Scatter Plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
 

# Extract the cluster assignments and feature values
cluster_labels = data['KMeans_Cluster']
x = data['Normalized_Recency']
y = data['Normalized_Frequency']
z = data['Normalized_MonetaryValue']
 

# Create the scatter plot
scatter = ax.scatter(x, y, z, c=cluster_labels, cmap='viridis', marker='o', s=50)

# Set the axis labels
ax.set_xlabel('Normalized_Recency')
ax.set_ylabel('Normalized_Frequency')
ax.set_zlabel('Normalized_MonetaryValue')
 

 # Add a title
ax.set_title('3D Scatter Plot of KMeans Clusters')
 

 # Add a legend
legend = ax.legend(*scatter.legend_elements(), loc="upper right", title="Clusters")
ax.add_artist(legend)
 

 # Show the plot
plt.show()


# 3. Box Plots (to see the distribution of each feature by cluster)
for feature in features:
 plt.figure(figsize=(8, 6))
 sns.boxplot(x='KMeans_Cluster', y=feature, data=data, palette='viridis')
 plt.title(f'{feature} by Cluster')
 plt.xlabel('Cluster')
 plt.ylabel(feature)
 plt.show()

 # --- Pie Chart Visualization ---

# Calculate cluster sizes
cluster_sizes = data['KMeans_Cluster'].value_counts()

# Create the pie chart
plt.figure(figsize=(8, 6))
plt.pie(cluster_sizes, labels=cluster_sizes.index, autopct='%1.1f%%', startangle=140, colors=plt.cm.viridis.colors)
plt.title('Distribution of Customers Across Clusters')
plt.show()
