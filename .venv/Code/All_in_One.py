import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load the Iris dataset
iris = load_iris()
X = iris.data  # Features (sepal length, sepal width, petal length, petal width)
y = iris.target  # Target labels (species)

# Create a DataFrame
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Add target column to the DataFrame
iris_df['target'] = iris.target

# Display the DataFrame
print(iris_df.head())

# Plot pairwise relationships in the iris dataset
sns.pairplot(iris_df, hue='target')
plt.show()

# Determine the optimal number of clusters using silhouette score
silhouette_scores = []
for n_clusters in range(2, 11):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)
    silhouette_scores.append(silhouette_avg)

# Determine the optimal number of clusters using Elbow Method
inertia = []
for n_clusters in range(2, 11):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_
    inertia.append(kmeans.inertia_)


# Plot silhouette scores
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs. Number of Clusters')
plt.grid(True)
plt.show()

# Plot Elbow Method
plt.plot(range(2, 11), inertia, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('within-cluster sum of squares (WCSS)')
plt.title(' Elbow Method : within-cluster sum of squares (WCSS) vs. Number of Clusters')
plt.grid(True)
plt.show()

fig, axes = plt.subplots(1, len(range(2,11)), figsize=(15, 5))

# Iterate through each number of clusters
for i, n_clusters in enumerate(range(2,11)):
    # Fit KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_

    # Plot the data points with colors representing the clusters
    axes[i].scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, alpha=0.5)

    # Plot the centroids
    axes[i].scatter(centroids[:, 0], centroids[:, 1], marker='x', s=200, color='red')

    axes[i].set_title(f"{n_clusters} clusters")

plt.tight_layout()
plt.show()