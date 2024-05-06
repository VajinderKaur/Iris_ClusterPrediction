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

# Determine the optimal number of clusters using Elbow Method
inertia = []
for n_clusters in range(2, 11):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_
    inertia.append(kmeans.inertia_)

# Plot Elbow Method
plt.plot(range(2, 11), inertia, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('within-cluster sum of squares (WCSS)')
plt.title(' Elbow Method : within-cluster sum of squares (WCSS) vs. Number of Clusters')
plt.grid(True)
plt.show()