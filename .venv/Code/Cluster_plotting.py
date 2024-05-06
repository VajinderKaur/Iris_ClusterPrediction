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