<h1>Optimal Number of Cluster Prediction for Iris Dataset </h1>
	
This repository contains code and visualizations for predicting the optimal number of clusters in the Iris dataset using unsupervised learning techniques. The project employs the Elbow method, Silhouette score, and pair plotting to determine the most suitable number of clusters for the dataset.

<h2>Methods Used:</h2>

**1.Elbow Method (Elbow_method.py)**: The Elbow method is a heuristic approach to find the optimal number of clusters based on the within-cluster sum of squares (WCSS). By plotting the WCSS against the number of clusters, the point where the rate of decrease sharply changes (the "elbow") suggests the optimal number of clusters.  
**2.Silhouette Score (Silhouette_score.py)**: The Silhouette score measures how similar an object is to its cluster compared to other clusters. It quantifies the quality of clustering by considering both the cohesion within clusters and the separation between clusters. This method aids in determining the optimal number of clusters by maximizing the Silhouette score.   
**3.Pair Plotting (Pair_plotting.py)**: Pair plotting is a visualization technique used to explore relationships between pairs of variables in a dataset. In this project, pair plotting is utilized to visualize the distribution of data points and assess the natural grouping present in the Iris dataset.  
**4.Cluster Plotting (Cluster_plotting.py)**: Cluster plotting involves visualizing the dataset with different numbers of clusters. By plotting data points with varying cluster numbers, this method provides insights into how the data is partitioned into clusters and helps identify the optimal number of clusters.

<h2>Contents</h2>

<h4>Code Folder</h4>Contains code for following:

**Elbow_method.py**: Code implementation of the Elbow method.   
**Silhouette_score.py**: Code implementation of the Silhouette score method.   
**Pair_plotting.py**: Code implementation for pair plotting visualization.    
**Cluster_plotting.py**: Code implementation for cluster plotting with varying numbers of clusters.   
**setup.py**: setup libraries required for the code to run!

<h4>Plots folder</h4>Contains visualizations generated during the analysis, including Elbow plots, Silhouette plots, pair plots, and cluster plots.

<h2>Results and Outputs:</h2>

All plots generated during the analysis, including Elbow plots, Silhouette plots, pair plots, and cluster plots for different cluster numbers, are saved in the "Plots" folder. Each method's code is stored in separate files for modularity and ease of understanding.

<h2>Conclusion</h2>

By employing multiple clustering techniques and visualization methods, this project offers a comprehensive analysis of the Iris dataset to determine the optimal number of clusters. The insights gained from these analyses can be valuable for various applications, including pattern recognition, data exploration, and predictive modeling.
