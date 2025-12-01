
# PACKAGE DESCRIPTION

`cluster_maker` is a Python package that provides a complete workflow for clustering analysis, covering data creation, preparation, algorithm execution, evaluation, and visualisation.  
Its design emphasises clarity, reliability, and a logical end-to-end process suitable for demonstrations, testing, and analytical work.

# Key Capabilities

- Generate synthetic datasets with defined cluster structures.  
- Explore and summarise data through basic statistical analysis.  
- Prepare data for clustering through feature selection and standardisation.  
- Perform K-means clustering using either a simple manual implementation or the scikit-learn version.  
- Assess clustering quality with standard evaluation metrics and diagnostic plots.  
- Visualise clustering results and inertia curves clearly.  
- Execute the full clustering workflow in a single high-level function.

# dataframe_builder.py

## Purpose  
Creates synthetic datasets with predefined cluster structures for controlled experimentation.

## Key Functions  
- **define_dataframe_structure(column_specs)** – constructs a template containing cluster centres and feature definitions.  
- **simulate_data(seed_df, n_points, cluster_std, random_state)** – generates data points around each centre with added noise and a `true_cluster` label.

# data_analyser.py

## Purpose  
Provides descriptive insights to support initial exploration of the dataset.

## Key Functions  
- **calculate_descriptive_statistics(data)** – returns standard summary metrics.  
- **calculate_correlation(data)** – generates a correlation matrix for numeric features.

# data_exporter.py

## Purpose  
Supports the clean and controlled export of processed data or results.

## Key Functions  
- **export_to_csv(data, filename, delimiter, include_index)** – saves data to a CSV file.  
- **export_formatted(data, file, include_index)** – writes a readable, well-formatted table to a text file.

# preprocessing.py

## Purpose  
Prepares datasets for clustering by validating inputs and ensuring all features share a comparable scale.

## Key Functions  
- **select_features(data, feature_cols)** – checks that chosen features exist and are numeric.  
- **standardise_features(X)** – standardises all features using `StandardScaler`.

# algorithms.py

## Purpose  
Implements K-means clustering both manually and through scikit-learn for comparison and practical use.

## Key Functions  
- **init_centroids(X, k)** – selects initial centroids.  
- **assign_clusters(X, centroids)** – assigns each point to its nearest centroid.  
- **update_centroids(X, labels, k)** – recalculates centroid positions.  
- **kmeans(X, k)** – runs the full manual K-means routine.  
- **sklearn_kmeans(X, k)** – executes the scikit-learn implementation.

# evaluation.py

## Purpose  
Provides metrics that assess clustering structure and performance.

## Key Functions  
- **compute_inertia(X, labels, centroids)** – calculates within-cluster variance.  
- **silhouette_score_sklearn(X, labels)** – computes silhouette scores to measure cohesion and separation.  
- **elbow_curve(X, k_values, use_sklearn)** – evaluates inertia across k values for elbow analysis.

# plotting_clustered.py

## Purpose  
Generates visualisations that clarify clustering behaviour and support interpretation.

## Key Functions  
- **plot_clusters_2d(X, labels, centroids, title)** – produces a 2D scatter plot of clusters.  
- **plot_elbow(k_values, inertias, title)** – visualises inertia values for selecting an appropriate number of clusters.

# interface.py

## Purpose  
Provides a single high-level function that performs the complete clustering workflow.

## Key Function  
- **run_clustering(...)**  
  Loads and prepares data, applies the selected clustering method, computes evaluation metrics, generates visualisations, and saves outputs.  
  Returns a structured dictionary containing labelled data, metrics, centroids, and generated figures.

# Summary

`cluster_maker` integrates data generation, preprocessing, clustering, evaluation, and visualisation into a cohesive and transparent framework.  
It supports both controlled experimentation with synthetic data and practical clustering analysis, offering a clear end-to-end process while still allowing detailed examination of each individual step.
