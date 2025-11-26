# cluster_maker – Package Description and Functional Overview

`cluster_maker` is an educational Python package designed for the MA52109
"Programming for Data Science" practical and mock exam at the University of Bath.
It provides a complete workflow for generating synthetic clustered data,
preprocessing it, running clustering algorithms, evaluating results, and producing
useful visualisations. The package is structured into clear, modular components
so that each stage of the analysis pipeline is easy to understand, test, and use.

This document describes the purpose of the package and explains the functionality
of its main parts.

---

## 1. Data Generation (`dataframe_builder.py`)

### **define_dataframe_structure(column_specs)**
Creates a template DataFrame representing cluster centres.
- `column_specs` is a list of dictionaries, each containing:
  - `"name"` — feature/column name  
  - `"reps"` — values representing the centre of each cluster  
- Produces a DataFrame with:
  - rows = clusters  
  - columns = features  

### **simulate_data(seed_df, n_points, cluster_std, random_state)**
Generates synthetic clustered data around the specified cluster centres.
- Uses multivariate Gaussian noise
- Ensures points are distributed evenly across clusters
- Returns a DataFrame containing:
  - all original feature columns  
  - a `true_cluster` column indicating the generating cluster  

This module is used heavily in demos and testing to create realistic synthetic datasets.

---

## 2. Data Analysis (`data_analyser.py`)

### **calculate_descriptive_statistics(data)**
Computes descriptive statistics (`count`, `mean`, `std`, `min`, `max`, etc.) for all numeric columns.

### **calculate_correlation(data)**
Computes the correlation matrix between all numeric variables.

These tools allow quick inspection and validation of datasets before clustering.

---

## 3. Data Export (`data_exporter.py`)

### **export_to_csv(data, filename, delimiter, include_index)**
Saves a DataFrame to a CSV file.

### **export_formatted(data, file, include_index)**
Exports a DataFrame as a neatly formatted table into a text file.

These functions support both machine-readable and human-readable outputs.

---

## 4. Preprocessing (`preprocessing.py`)

### **select_features(data, feature_cols)**
Selects user-specified numeric columns from a DataFrame.
- Validates column names  
- Ensures selected columns are numeric  

### **standardise_features(X)**
Standardises features to zero mean and unit variance using `StandardScaler`.

These operations prepare data for clustering by putting all variables on comparable scales.

---

## 5. Clustering Algorithms (`algorithms.py`)

This module provides two implementations of K-means:

### **kmeans(X, k, max_iter, tol, random_state)**
A fully manual implementation of the K-means algorithm.
Includes:
- random centroid initialisation  
- cluster assignment  
- centroid update  
- stopping tolerance  

### **sklearn_kmeans(X, k, random_state)**
A thin wrapper around `sklearn.cluster.KMeans`.

### Additional utilities:
- **init_centroids** – random centroid selection  
- **assign_clusters** – assigns each point to nearest centroid  
- **update_centroids** – recomputes cluster means  

---

## 6. Evaluation (`evaluation.py`)

### **compute_inertia(X, labels, centroids)**
Computes the sum of squared distances within each cluster.

### **silhouette_score_sklearn(X, labels)**
Uses sklearn to compute the silhouette score.

### **elbow_curve(X, k_values, random_state, use_sklearn)**
Computes inertia for multiple values of *k* for use in an elbow plot.

These metrics help judge the quality and stability of clustering results.

---

## 7. Visualisation (`plotting_clustered.py`)

### **plot_clusters_2d(X, labels, centroids, title)**
Creates a 2D scatter plot of clustered data (first two features).

### **plot_elbow(k_values, inertias, title)**
Plots inertia vs number of clusters.

These plots help users interpret clustering structure and optimal k-values.

---

## 8. High-Level Pipeline (`interface.py`)

### **run_clustering(...)**
A complete end-to-end workflow that:
1. Loads a CSV file  
2. Selects and standardises features  
3. Runs K-means  
4. Computes inertia and silhouette score  
5. Generates plots  
6. Optionally exports labelled data  

It returns a dictionary containing:
- labelled DataFrame  
- cluster labels  
- centroids  
- metrics  
- plots  
- elbow curve results (optional)  

This function enables beginners to perform clustering with a single call.

---

## 9. Demo Scripts (`demo/`)

The demo scripts illustrate how to use the package end-to-end.

The main demo:
- generates synthetic clustered data,
- runs the full analysis pipeline,
- saves plots and outputs into `demo_output/`.

It is intended for teaching and to provide students with a working example of the workflow.

---

## 10. Overall Structure and Purpose

`cluster_maker` is intentionally modular and readable, enabling students to:
- understand each computational step,
- diagnose and fix errors,
- extend the codebase with new techniques,
- write tests and demos,
- produce reproducible clustering analyses.

The package demonstrates good software engineering practices for data science:
clear interfaces, modular design, separation of concerns, and practical plotting and evaluation tools.

---

# End of DESCRIPTION.md
