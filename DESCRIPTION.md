# cluster_maker Package Description

The `cluster_maker` package is an educational tool for simulating and analysing clustered data.

## Key Modules
* **dataframe_builder**: Generates synthetic data points around specified cluster centres.
* **preprocessing**: Handles feature selection and standardisation (scaling).
* **algorithms**: Contains the K-Means clustering logic.
* **evaluation**: Calculates metrics like Inertia and Silhouette score.
* **plotting_clustered**: Visualises the data and the "Elbow Curve".
* **data_analyser**: Calculates basic statistics.
* **data_exporter**: Saves results to CSV or text files.

## Workflow
1. Define cluster centres.
2. Simulate data points.
3. Preprocess (Scale) the data.
4. Apply K-Means algorithm.
5. Evaluate and Plot results.