I have used the cluster_maker to generate the data (my_data.csv) through the file 'Generating_data.py'.

This 'my_data.csv' is then used in the cluster_analysis.py file

When we run the file in demo it produces a directory called demo_output.
demo_output contains:
- cluster_plot.png --> This plots the clusters
- clustered_data.csv 
- elbow_plot.png --> This gives an elbow plot.


Description of cluster_maker

1) The cluster_maker has an algorithms.py module that  implements a full manual K-means algorithm with functions for initialising centroids, assigning clusters, and updating centroids. It runs iterative updates until convergence while handling empty clusters safely. It also includes a thin wrapper around scikit-learn’s KMeans to return labels and centroids.
2) The data_analysis.py file provides two utility functions for basic data analysis on pandas DataFrames. It computes descriptive statistics for numeric columns and generates a correlation matrix. Both functions validate the input and then return the corresponding pandas calculations.
3) The data_exporter.py file provides functions for exporting pandas DataFrames either as CSV files or as neatly formatted text tables. It validates that the input is a DataFrame before writing the output to a file or file-like object. 
4) The dataframe_builder.py file defines a function that builds a structured DataFrame describing cluster centres from user-provided specifications. It also provides a data simulation function that generates synthetic clustered data by adding Gaussian noise around those centres. Together, they allow users to create controlled cluster structures and simulate labelled datasets for experimentation.
5) The evaluation.py file provides utilities for evaluating clustering quality, including computing inertia (the within-cluster sum of squared distances) and calculating silhouette scores using scikit-learn. It also implements an elbow-curve function that runs K-means for multiple values of k and records the corresponding inertia values. Overall, it supports analysis and comparison of clustering performance across different cluster counts.
6) The interface.py file defines a high-level workflow that loads data, selects and optionally standardises features, and runs either manual or scikit-learn K-means clustering. It then computes evaluation metrics such as inertia and silhouette score, adds cluster labels back to the dataset, and generates plots for clusters and (optionally) an elbow curve. The function returns all results—including labels, centroids, metrics, plots, and optionally exported data—in a single dictionary.
7) The plotting_clustered.py file defines two plotting utilities for visualising clustering results. One function produces a 2D scatter plot of data coloured by cluster labels, optionally marking centroids, while the other generates an elbow plot showing inertia values across different choices of k. Together they provide visual tools for interpreting cluster assignments and selecting an appropriate number of clusters.
8) The preprocessing.py file provides utilities for preparing data before clustering. It includes a function that selects specified feature columns from a DataFrame and ensures they are all numeric, and another that standardises feature arrays to zero mean and unit variance. Together they handle feature selection and preprocessing for downstream clustering workflows.


