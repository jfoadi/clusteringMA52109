# cluster_maker - Clustering Analysis Package

## Overview
cluster_maker is an educational Python package designed for generating synthetic clustered data, performing clustering analysis, and visualizing results. It provides a complete workflow from data generation to clustering evaluation, making it ideal for practical exercises and educational purposes.

## Package Structure

### Main Components
#### 1. **Clustering Algorithms** (`algorithms.py`)
- **Custom K-means Implementation**:
  - `init_centroids()`: Randomly initializes cluster centers
  - `assign_clusters()`: Assigns points to nearest centroids using Euclidean distance
  - `update_centroids()`: Updates centroids based on cluster means
  - `kmeans()`: Complete custom K-means algorithm with convergence checking - **Scikit-learn Integration**:
  - `sklearn_kmeans()`: Wrapper around scikit-learn's KMeans implementation


#### 2. **Data Analysis & Preprocessing** (`data_analyser.py`, `preprocessing.py`)
- **`calculate_descriptive_statistics()`**: Computes basic statistics for numeric columns
- **`calculate_correlation()`**: Generates correlation matrices for feature analysis
- **`select_features()`**: Validates and selects appropriate numeric columns for clustering
- **`standardise_features()`**: Normalizes feature data using scikit-learn's StandardScaler
#### 3. **Data Export** (`data_exporter.py`)
- **`export_to_csv()`**: Saves DataFrames to CSV format
- **`export_formatted()`**: Exports data as formatted text tables

#### 4. **Data Generation & Simulation** (`dataframe_builder.py`)
- **`define_dataframe_structure()`**: Creates seed DataFrames defining cluster centers with specified feature distributions
- **`simulate_data()`**: Generates synthetic clustered data points around specified centers with configurable Gaussian noise


#### 4. **Clustering Evaluation** (`evaluation.py`)
- **`compute_inertia()`**: Calculates within-cluster sum of squares
- **`silhouette_score_sklearn()`**: Measures cluster quality using silhouette scores
- **`elbow_curve()`**: Computes inertia values for multiple k values to determine optimal clusters

#### 5. **High-Level Interface** (`interface.py`)
- **`run_clustering()`**: Main orchestrator function that coordinates the entire clustering pipeline from data loading to result visualization
#### 6. **Visualization** (`plotting_clustered.py`)
- **`plot_clusters_2d()`**: Creates 2D scatter plots of clustered data with centroid markers
- **`plot_elbow()`**: Generates elbow method plots for optimal k selection

#### 7. **High-Level Interface** (`interface.py`)
- **`run_clustering()`**: Main orchestrator function that coordinates the entire clustering pipeline from data loading to result visualization

## Key Features

- **End-to-End Workflow**: Complete pipeline from data generation to clustering evaluation
- **Dual Implementations**: Both custom and scikit-learn clustering algorithms
- **Comprehensive Evaluation**: Multiple metrics including inertia and silhouette scores
- **Professional Visualization**: 2D cluster plots and elbow method analysis
- **Educational Focus**: Designed for practical exercises with clear, modular code
- **Data Flexibility**: Support for various input formats and feature selection
- **Reproducible Results**: Random state control for consistent outputs

## Supported Libraries
- Python Standard Library
- NumPy
- pandas
- matplotlib
- SciPy
- scikit-learn

## Typical Workflow
1. Generate or load clustered data
2. Select and preprocess features
3. Run clustering algorithms (custom or scikit-learn)
4. Evaluate results using multiple metrics
5. Visualize clusters and determine optimal parameters
6. Export results for further analysis

The package is particularly suited for educational settings where students can explore clustering concepts through both custom implementations and established library functions.