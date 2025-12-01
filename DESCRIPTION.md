# cluster_maker: Educational Clustering Package

## Overview

`cluster_maker` is an educational Python package designed for generating synthetic clustered data, performing clustering analysis, evaluating results, and producing informative visualizations. The package serves as a practical tool for students to understand clustering algorithms and data analysis workflows through hands-on implementation and experimentation.

## Structure of cluster_maker Package

cluster_maker/ # Main package directory
├── init.py # Package initialization and exports
├── algorithms.py # Clustering algorithm implementations
├── data_analyser.py # Statistical analysis functions
├── data_exporter.py # Data export utilities
├── dataframe_builder.py # Synthetic data generation
├── evaluation.py # Cluster evaluation metrics
├── interface.py # High-level workflow orchestration
├── plotting_clustered.py # Visualization functions
└── preprocessing.py # Data preprocessing utilities

## Functionality of its Main Components

### 1. Data Generation & Structure
**Module: `dataframe_builder.py`**
- **`define_dataframe_structure()`**: Creates a seed DataFrame defining cluster centers with specified feature columns and their representative values across clusters
- **`simulate_data()`**: Generates synthetic data points around cluster centers with Gaussian noise, automatically distributing points across clusters and adding true cluster labels

### 2. Data Analysis & Exploration
**Module: `data_analyser.py`**
- **`calculate_descriptive_statistics()`**: Computes comprehensive statistical summaries (count, mean, std, min/max, quartiles) for numeric columns
- **`calculate_correlation()`**: Generates correlation matrices to identify relationships between numeric features

### 3. Data Preprocessing
**Module: `preprocessing.py`**
- **`select_features()`**: Validates and selects specific numeric columns for clustering, ensuring data integrity
- **`standardise_features()`**: Standardizes features to zero mean and unit variance using scikit-learn's StandardScaler for improved clustering performance

### 4. Clustering Algorithms
**Module: `algorithms.py`**
- **Manual Implementation**:
  - **`init_centroids()`**: Randomly initializes cluster centers from data points
  - **`assign_clusters()`**: Assigns points to nearest centroids using Euclidean distance
  - **`update_centroids()`**: Recalculates centroids as cluster means, handling empty clusters
  - **`kmeans()`**: Complete manual K-means implementation with convergence checking
- **Scikit-learn Integration**:
  - **`sklearn_kmeans()`**: Wrapper around scikit-learn's optimized KMeans implementation

### 5. Cluster Evaluation
**Module: `evaluation.py`**
- **`compute_inertia()`**: Calculates within-cluster sum of squared distances (inertia)
- **`silhouette_score_sklearn()`**: Computes silhouette scores for cluster quality assessment
- **`elbow_curve()`**: Generates inertia values across multiple K values for optimal cluster number selection

### 6. Visualization
**Module: `plotting_clustered.py`**
- **`plot_clusters_2d()`**: Creates 2D scatter plots of clustered data with centroid markers and color-coded clusters
- **`plot_elbow()`**: Generates elbow curves to help determine optimal number of clusters

### 7. Data Export
**Module: `data_exporter.py`**
- **`export_to_csv()`**: Exports DataFrames to CSV format with customizable delimiters
- **`export_formatted()`**: Creates formatted text table outputs for readable data presentation

### 8. High-Level Interface
**Module: `interface.py`**
- **`run_clustering()`**: Orchestrates the complete clustering workflow:
  - Loads data from CSV
  - Selects and preprocesses features
  - Runs chosen clustering algorithm
  - Computes evaluation metrics
  - Generates visualizations
  - Exports results

## Educational Value

`cluster_maker` is specifically designed for educational purposes, providing:
- **Transparent implementations** of clustering algorithms
- **Comprehensive workflow** from data generation to evaluation
- **Multiple algorithm options** for comparison learning
- **Visual feedback** through automated plotting
- **Robust error handling** for learning proper coding practices


## Dependencies

- Python Standard Library
- numpy
- pandas  
- matplotlib
- scipy
- scikit-learn
