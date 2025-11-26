# cluster_maker Package Description

## Overview
`cluster_maker` is an educational Python package designed for generating synthetic clustered data, running clustering algorithms, evaluating results, and producing user-friendly visualizations. The package is intended for practicals and exams where students work with and extend the clustering workflow.

## Main Components

### 1. Data Generation (`dataframe_builder.py`)
- **`define_dataframe_structure(column_specs)`**: Creates a seed DataFrame that defines cluster centres. Each row represents a cluster and each column represents a feature dimension.
- **`simulate_data(seed_df, n_points, cluster_std, random_state)`**: Generates synthetic clustered data by adding Gaussian noise around cluster centres. Returns a DataFrame with the original features plus a `true_cluster` column indicating the true cluster assignment.

### 2. Data Analysis (`data_analyser.py`)
- **`calculate_descriptive_statistics(data)`**: Computes descriptive statistics (count, mean, std, min, max, etc.) for numeric columns in a DataFrame.
- **`calculate_correlation(data)`**: Computes the correlation matrix for numeric columns in a DataFrame.

### 3. Data Export (`data_exporter.py`)
- **`export_to_csv(data, filename, delimiter, include_index)`**: Exports a DataFrame to a CSV file with configurable delimiter and index options.
- **`export_formatted(data, file, include_index)`**: Exports a DataFrame as a human-readable formatted text table to a file or file-like object.

### 4. Preprocessing (`preprocessing.py`)
- **`select_features(data, feature_cols)`**: Selects a subset of numeric columns from a DataFrame. Validates that all requested columns exist and are numeric.
- **`standardise_features(X)`**: Standardises features to zero mean and unit variance using scikit-learn's StandardScaler.

### 5. Clustering Algorithms (`algorithms.py`)
- **`init_centroids(X, k, random_state)`**: Initialises k centroids by randomly sampling points from the data.
- **`assign_clusters(X, centroids)`**: Assigns each data point to the nearest centroid using Euclidean distance.
- **`update_centroids(X, labels, k, random_state)`**: Updates centroids as the mean of points in each cluster. Handles empty clusters by re-initialising them randomly.
- **`kmeans(X, k, max_iter, tol, random_state)`**: Manual implementation of the K-means clustering algorithm.
- **`sklearn_kmeans(X, k, random_state)`**: Wrapper around scikit-learn's KMeans for comparison.

### 6. Evaluation (`evaluation.py`)
- **`compute_inertia(X, labels, centroids)`**: Computes the within-cluster sum of squared distances (inertia).
- **`silhouette_score_sklearn(X, labels)`**: Computes the silhouette score using scikit-learn.
- **`elbow_curve(X, k_values, random_state, use_sklearn)`**: Computes inertia values for multiple k values to support the elbow method for selecting optimal k.

### 7. Plotting (`plotting_clustered.py`)
- **`plot_clusters_2d(X, labels, centroids, title)`**: Plots clustered data in 2D using the first two features, with optional centroid markers.
- **`plot_elbow(k_values, inertias, title)`**: Plots the elbow curve (inertia vs k).

### 8. High-Level Interface (`interface.py`)
- **`run_clustering(input_path, feature_cols, algorithm, k, standardise, output_path, random_state, compute_elbow, elbow_k_values)`**: High-level orchestration function that:
  - Loads data from a CSV file
  - Selects and optionally standardises features
  - Runs the chosen clustering algorithm
  - Computes evaluation metrics (inertia and silhouette score)
  - Generates 2D cluster plots and optionally elbow plots
  - Exports results to CSV and returns a comprehensive result dictionary

## Allowed Dependencies
- Python standard library
- NumPy
- Pandas
- Matplotlib
- SciPy
- Scikit-learn

## Usage Example
```python
from cluster_maker import run_clustering

result = run_clustering(
    input_path="data.csv",
    feature_cols=["x", "y"],
    algorithm="sklearn_kmeans",
    k=3,
    standardise=True,
    output_path="clustered_output.csv",
    random_state=42,
    compute_elbow=True
)

# Access results
print(result["metrics"])
result["fig_cluster"].show()
```

## Key Features
- **Modular design**: Each component can be used independently or as part of the full workflow
- **Educational focus**: Clear, well-documented code suitable for learning clustering concepts
- **Flexibility**: Supports custom feature selection, multiple algorithms, and various evaluation metrics
- **Visualization**: Built-in plotting functions for understanding clustering results
- **Error handling**: Comprehensive input validation and informative error messages
