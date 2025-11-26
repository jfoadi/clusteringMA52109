# cluster_maker Package Description

## Overview

`cluster_maker` is an educational Python package designed for generating synthetic clustered data, running clustering algorithms, evaluating clustering results, and producing visualizations. The package is intended for teaching and practical work where students learn to work with clustering analysis.

## Main Purpose

The package provides a complete workflow for cluster analysis:
1. **Generate** synthetic clustered datasets for testing and learning
2. **Preprocess** data through feature selection and standardization
3. **Apply** clustering algorithms (K-means)
4. **Evaluate** clustering quality using standard metrics
5. **Visualize** results with 2D scatter plots and elbow curves
6. **Export** results to CSV files

---

## Main Components

### 1. Data Generation (`dataframe_builder.py`)

**Purpose**: Create synthetic datasets with known cluster structures for testing and education.

**Key Functions**:
- `define_dataframe_structure(column_specs)` - Defines cluster centers from column specifications
- `simulate_data(seed_df, n_points, cluster_std, random_state)` - Generates random points around cluster centers with Gaussian noise

**Use Case**: Creating reproducible test datasets with controlled cluster separation.

---

### 2. Data Analysis (`data_analyser.py`)

**Purpose**: Perform basic statistical analysis on datasets.

**Key Functions**:
- `calculate_descriptive_statistics(df)` - Computes summary statistics (mean, std, min, max)
- `calculate_correlation(df)` - Calculates correlation matrices for numeric features
- `calculate_numeric_summary(df)` - Computes detailed numeric summary including missing values

**Use Case**: Understanding data distributions before clustering.

---

### 3. Data Export (`data_exporter.py`)

**Purpose**: Save analysis results in various formats.

**Key Functions**:
- `export_to_csv(df, filepath)` - Exports DataFrames to CSV format
- `export_formatted(df, filepath)` - Exports data in human-readable text format
- `export_numeric_summary(df, csv_path, text_path)` - Exports summary to both CSV and text formats

**Use Case**: Sharing results and creating reports.

---

### 4. Preprocessing (`preprocessing.py`)

**Purpose**: Prepare data for clustering algorithms.

**Key Functions**:
- `select_features(df, feature_cols)` - Extracts specified numeric columns for clustering
- `standardise_features(X)` - Standardizes features to zero mean and unit variance
- `apply_pca(df, n_components)` - Applies Principal Component Analysis (PCA) to reduce dimensionality

**Use Case**: Ensuring features are on comparable scales to prevent bias in distance calculations.

---

### 5. Clustering Algorithms (`algorithms.py`)

**Purpose**: Implement K-means clustering algorithms.

**Key Functions**:
- `kmeans(X, k, max_iter, random_state)` - Custom K-means implementation from scratch
- `sklearn_kmeans(X, k, random_state)` - Wrapper for scikit-learn's K-means
- `init_centroids(X, k, random_state)` - Random centroid initialization
- `assign_clusters(X, centroids)` - Assigns each point to nearest centroid
- `update_centroids(X, labels, k)` - Recalculates centroids as cluster means

**Use Case**: Educational demonstration of K-means algorithm internals and production-ready clustering.

---

### 6. Evaluation (`evaluation.py`)

**Purpose**: Assess clustering quality using standard metrics.

**Key Functions**:
- `compute_inertia(X, labels, centroids)` - Calculates within-cluster sum of squares (inertia/WCSS)
- `silhouette_score_sklearn(X, labels)` - Computes silhouette coefficient (-1 to +1, higher is better)
- `elbow_curve(X, k_range, algorithm)` - Generates inertia values for different k values

**Use Case**: Determining optimal number of clusters and validating cluster quality.

---

### 7. Visualization (`plotting_clustered.py`)

**Purpose**: Create informative plots of clustering results.

**Key Functions**:
- `plot_clusters_2d(X, labels, centroids, feature_names)` - Creates scatter plot with colored clusters
- `plot_elbow(k_values, inertias)` - Generates elbow curve plot for k selection

**Use Case**: Visual inspection of clustering results and k selection.

---

### 8. High-Level Interface (`interface.py`)

**Purpose**: Orchestrate the complete clustering workflow in a single function call.

**Key Function**:
- `run_clustering(..., use_pca=False, n_components=2, ...)` - End-to-end clustering pipeline (now supports PCA)

**Workflow**:
1. Loads CSV data
2. Selects and validates features
3. Optionally standardizes features
4. Runs clustering algorithm
5. Computes evaluation metrics (inertia, silhouette score)
6. Optionally computes elbow curve
7. Creates visualizations
8. Saves results to CSV

**Returns**: Dictionary containing clustered data, metrics, and matplotlib figures

**Use Case**: Single-function interface for complete clustering analysis, ideal for demos and scripts.

---

## Typical Workflow

```python
from cluster_maker import run_clustering

# Run complete clustering pipeline
result = run_clustering(
    input_path='data.csv',
    feature_cols=['x', 'y'],
    algorithm='kmeans',
    k=3,
    standardise=True,
    output_path='output/clusters.csv',
    random_state=42,
    compute_elbow=True
)

# Access results
print(result['metrics'])  # {'inertia': ..., 'silhouette': ...}
result['fig_cluster'].savefig('cluster_plot.png')
result['fig_elbow'].savefig('elbow_plot.png')
```

## Allowed Dependencies

The package uses only standard data science libraries:
- **Python standard library** - Core Python functionality
- **numpy** - Numerical computations
- **pandas** - Data manipulation and CSV I/O
- **matplotlib** - Plotting and visualization
- **scipy** - Scientific computing utilities
- **scikit-learn** - Machine learning algorithms and metrics
