I have used the cluster_maker to generate the data (,y_data.csv) through the file 'Generating_data.py'.

This 'my_data.csv' is then used in the cluster_analysis.py file

Description of cluster_maker

1) The cluster_maker has an algorithms.py module that  implements a full manual K-means algorithm with functions for initialising centroids, assigning clusters, and updating centroids. It runs iterative updates until convergence while handling empty clusters safely. It also includes a thin wrapper around scikit-learn’s KMeans to return labels and centroids.
2) The data_analysis.py file provides two utility functions for basic data analysis on pandas DataFrames. It computes descriptive statistics for numeric columns and generates a correlation matrix. Both functions validate the input and then return the corresponding pandas calculations.
3) The data_exporter.py file provides functions for exporting pandas DataFrames either as CSV files or as neatly formatted text tables. It validates that the input is a DataFrame before writing the output to a file or file-like object. 
4) The dataframe_builder.py file defines a function that builds a structured DataFrame describing cluster centres from user-provided specifications. It also provides a data simulation function that generates synthetic clustered data by adding Gaussian noise around those centres. Together, they allow users to create controlled cluster structures and simulate labelled datasets for experimentation.
5) The evaluation.py file provides utilities for evaluating clustering quality, including computing inertia (the within-cluster sum of squared distances) and calculating silhouette scores using scikit-learn. It also implements an elbow-curve function that runs K-means for multiple values of k and records the corresponding inertia values. Overall, it supports analysis and comparison of clustering performance across different cluster counts.
6) The interface.py file defines a high-level workflow that loads data, selects and optionally standardises features, and runs either manual or scikit-learn K-means clustering. It then computes evaluation metrics such as inertia and silhouette score, adds cluster labels back to the dataset, and generates plots for clusters and (optionally) an elbow curve. The function returns all results—including labels, centroids, metrics, plots, and optionally exported data—in a single dictionary.
7) 
