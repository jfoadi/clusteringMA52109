`cluster_maker` is a small educational Python package for simulating clustered
datasets and running clustering analyses with a simple, user-friendly interface.
The package is built upon standard scientific libraries including NumPy, pandas, and scikit-learn.

The package's functionality is partitioned into distinct modules:

dataframe_builder.py: Defines the cluster center structure (seed DataFrame) and simulates large, clustered datasets by adding Gaussian noise around these centers.

data_analyser.py: Computes basic descriptive statistics and correlation matrices for the dataset's features.

data_exporter.py: Facilitates the export of processed data to CSV files and formatted results to text reports.

preprocessing.py: Handles essential preprocessing steps, including feature selection and standardisation (scaling) of numerical features prior to clustering.

algorithms.py: Implements both a simple, manual K-means algorithm and a robust scikit-learn KMeans wrapper for cluster assignment and centroid calculation.

evaluation.py: Calculates key metrics to assess cluster quality, including Inertia (within-cluster sum of squares) and the Silhouette Score. It also computes data for the Elbow Curve.

plotting_clustered.py: Generates and saves graphical outputs, including 2D scatter plots showing clustered data with centroids, and plots of the Elbow Curve.

interface.py: Contains the main run_clustering function, which serves as the package's single, user-friendly entry point, orchestrating the entire workflow from input to output.

generate_data: Creates a simulated dataset for use by the demo file.