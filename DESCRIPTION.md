# cluster_maker – Package Description

`cluster_maker` is a modular Python package created for the module  
**MA52109 – Programming for Data Science** at the University of Bath.

It provides a complete workflow for:

- generating synthetic clustered datasets,
- preprocessing and analysing data,
- applying clustering algorithms (manual and scikit-learn),
- evaluating cluster quality,
- visualising results,
- coordinating all steps through a high-level interface.

The package is intentionally organised into small modules so students can easily
understand, debug, and extend each component during practicals and exams.
## 1. Data Generation – `dataframe_builder.py`

The module `dataframe_builder.py` provides tools for defining cluster centres
and generating synthetic datasets around them. It is the starting point for
building clustered data used throughout the package.

---

### **`define_dataframe_structure(column_specs)`**

This function creates a DataFrame where:

- each **row** corresponds to a cluster centre,
- each **column** corresponds to a feature.

`column_specs` must be a list of dictionaries, each containing:
- `"name"` → feature name  
- `"reps"` → list of representative values, one per cluster  

Example:
```python
{"name": "x", "reps": [0, 5, -5]}

If the number of representative values ("reps") is inconsistent across features,
the function raises a clear "ValueError" to ensure the structure of the seed
DataFrame is valid.



### **`simulate_data(seed_df, n_points, cluster_std, random_state)`**

This function generates a synthetic dataset by creating points around the cluster
centres defined in "seed_df".

The process works as follows:

- Data points are distributed as evenly as possible across clusters.
- For each cluster centre, Gaussian noise with standard deviation "cluster_std"
  is added to generate realistic variation.
- A new column, "true_cluster", is added to the output DataFrame to record
  which cluster each point originated from.

The function returns a complete DataFrame containing all generated feature
columns plus the "true_cluster" label for evaluation and comparison.

## 2. Exploratory Analysis – `data_analyser.py`

The module "data_analyser.py" provides simple but essential tools for exploring
and understanding the dataset before applying clustering algorithms.

-

### **calculate_descriptive_statistics(df)**

This function computes standard summary statistics for all **numeric** columns
in the DataFrame, including:

- count,
- mean,
- standard deviation,
- minimum and maximum,
- quartiles (25%, 50%, 75%).

It is useful for obtaining a quick overview of the data distribution and
identifying potential irregularities.



### **calculate_correlation(df)**

Computes and returns the **correlation matrix** between all numeric columns,
using Pearson correlation.

This helps detect relationships or dependencies between variables, which can
be important before clustering. Non-numeric columns are automatically ignored.

## 3. Data Export – `data_exporter.py`

The module "data_exporter.py" provides functions for saving results in both
machine-readable and human-readable formats. This is especially useful when
producing outputs for reports, analysis pipelines, or user-facing summaries.


### **`export_to_csv(df, filename, delimiter=",", include_index=False)`**

Exports a pandas DataFrame to a CSV file.

Key features:
- allows specifying a custom delimiter (default: comma),
- allows including or excluding the DataFrame index,
- raises a clear error if the input is not a valid DataFrame.

This function is typically used to save:
- clustered datasets,
- processed data,
- summary statistics.


### **`export_formatted(df, file, include_index=False)`**

Writes a human-readable, text-formatted table based on the DataFrame’s content.

Characteristics:
- accepts either a **file path** or an **open file handle**,
- uses "DataFrame.to_string()" to produce a neatly aligned text table,
- useful for readable summaries or logs.

This function is designed for users who need a clean textual representation
of the results, without relying on spreadsheet software.


## 4. Preprocessing – `preprocessing.py`

The module "preprocessing.py" contains essential tools for preparing data before
clustering. Proper preprocessing ensures that clustering algorithms operate on
clean, numeric, and well-scaled data.


### **`select_features(df, feature_cols)`**

Selects specific columns from a DataFrame and ensures they are valid for
clustering.

This function performs three checks:

1. **Existence check**  
   Ensures all requested feature names are present in the DataFrame.  
   Missing columns trigger a clear "KeyError".

2. **Type check**  
   Ensures that selected features are numeric.  
   Non-numeric columns raise a descriptive "TypeError".

3. **Safe return**  
   Returns a copy of the selected numeric columns only.

This prevents clustering algorithms from being applied to invalid or mixed data types.


### **`standardise_features(X)`**

Standardises the numeric feature matrix using scikit-learn’s "StandardScaler".

Each feature is transformed to have:

- mean = 0  
- standard deviation = 1  

Standardisation is crucial because clustering algorithms like K-means are
distance-based and are heavily influenced by the scale of the variables.

The function returns the scaled NumPy array, ready for clustering.

## 5. Clustering Algorithms – `algorithms.py`

The module "algorithms.py" contains both a **custom implementation of the K-means
algorithm** and a **wrapper around scikit-learn’s KMeans**.  
This dual approach allows users to understand how clustering works internally
and also to rely on a robust, optimised library implementation.


### **Manual K-means Components**

The package breaks the K-means algorithm into clear, modular steps, making the
logic transparent and easier to debug.


#### **`init_centroids(X, k, random_state=None)`**
Randomly selects "k" points from the dataset "X" to serve as initial centroids.
If "k" exceeds the number of samples, a descriptive error is raised.


#### **`assign_clusters(X, centroids)`**
Assigns each data point to the nearest centroid using Euclidean distance.

Returns:
- an array of cluster labels for each sample.


#### **`update_centroids(X, labels, k, random_state=None)`**
Recomputes each centroid as the mean of all points assigned to that cluster.

Special handling:
- If a cluster becomes empty, a new centroid is randomly reinitialised from the data.


#### **`kmeans(X, k, max_iter=300, tol=1e-4, random_state=None)`**
Runs the full K-means loop:

1. initialisation,  
2. cluster assignment,  
3. centroid update,  
4. convergence check.

Returns:
- an array of cluster labels,
- the final centroids.

This implementation makes the algorithm’s mechanics fully transparent.


### **Scikit-learn Wrapper**

#### **`sklearn_kmeans(X, k, random_state=None)`**
A thin wrapper around "sklearn.cluster.KMeans".

- Runs scikit-learn’s K-means algorithm,
- Uses "n_init=10" for stability,
- Returns labels and cluster centres.

This provides a reliable and fast alternative to the manual implementation.

## 6. Evaluation – `evaluation.py`

The module "evaluation.py" provides tools for assessing the quality and structure
of clustering solutions. These functions help users compare different models,
choose optimal parameters, and understand how well the algorithm has performed.


### **`compute_inertia(X, labels, centroids)`**

Computes the **within-cluster sum of squared distances**, also known as inertia.

How it works:
- For each point, it measures the squared distance to its assigned centroid.
- These distances are summed across all samples.

Inertia is a core metric for evaluating how compact the clusters are.  
Lower inertia indicates tighter, more coherent clusters.


### **`silhouette_score_sklearn(X, labels)`**

Computes the **silhouette score** using scikit-learn.

Silhouette score:
- measures how similar each point is to its assigned cluster compared to others,
- ranges from **-1 to 1**,
- higher values indicate better separation between clusters.

If the clustering contains fewer than two clusters, a descriptive error is raised.


### **`elbow_curve(X, k_values, random_state=None, use_sklearn=True)`**

Computes inertia values across multiple values of "k" to support the  
**elbow method** for choosing the best number of clusters.

Returns:
- a dictionary mapping each value of "k" to its computed inertia.

The user can choose whether to run:
- the manual K-means implementation, or
- scikit-learn’s KMeans (default).

## 7. Visualisation – `plotting_clustered.py`

The module "plotting_clustered.py" provides visual tools to help users interpret
and communicate clustering results. Visualisations are essential for understanding
cluster separation, structure, and the effect of choosing different values of "k".


### **`plot_clusters_2d(X, labels, centroids=None, title=None)`**

Creates a "2D" scatter plot using the **first two features** of the dataset.

Key features:
- points are coloured by their assigned cluster label,
- optional centroid markers are plotted with large black "×" symbols,
- includes axis labels, a colourbar, and an optional title.

This plot is particularly useful for visually inspecting cluster separation
and centroid positions.


### **`plot_elbow(k_values, inertias, title="Elbow Curve")`**

Plots inertia against the number of clusters "k".

Characteristics:
- each point is plotted with a circle marker,
- a connecting line helps reveal the “elbow”,
- includes axis labels, grid, and a clear title.

This visualisation supports the **elbow method**, helping users choose an
appropriate number of clusters based on inertia behaviour.

## 8. High-Level Interface – `interface.py`

The module "interface.py" provides a single, user-friendly function that ties
together all components of the package.  
It is intended to be the main entry point for end users and scripts.


### **`run_clustering(input_path, feature_cols, algorithm="kmeans", k=3, standardise=True, output_path=None, random_state=None, compute_elbow=False, elbow_k_values=None)`**

This function orchestrates the **entire clustering workflow**, performing all
steps automatically:

1. **Load the input CSV file** using pandas.  
2. **Select the requested feature columns** via "select_features()".  
3. **Standardise the features** (optional) using "standardise_features()".  
4. **Run a clustering algorithm**, choosing between:
   - the manual K-means implementation, or  
   - scikit-learn’s "KMeans".  
5. **Compute evaluation metrics**, including:
   - inertia,  
   - silhouette score (if applicable).  
6. **Generate visualisations**, including:
   - a "2D" cluster scatter plot,  
   - optionally an elbow plot (if "compute_elbow=True").  
7. **Export the clustered dataset** to CSV if an output path is provided.

The function returns a dictionary containing:

- "data" — the input DataFrame with an added `"cluster"` column,  
- "labels" — cluster assignments,  
- "centroids" — final cluster centres,  
- "metrics" — inertia and silhouette score,  
- "fig_cluster" — the cluster scatter plot figure,  
- "fig_elbow" — the elbow plot figure (or "None"),  
- "elbow_inertias" — a dictionary "k → inertia" (if computed).

This high-level function allows users to run a complete clustering analysis with
a single call and without interacting directly with lower-level modules.

## 9. Demo Script – `demo/cluster_analysis.py`

The demo script illustrates how a user interacts with the "cluster_maker" package
from the command line.  
It is designed to be simple, informative, and suitable for non-expert users.

The script performs the following actions:

1. **Accepts a CSV file path** from the command line.  
2. **Validates the input** and prints clear error messages for incorrect usage.  
3. **Loads the dataset** and automatically identifies numeric columns.  
4. **Selects two numeric features** for a "2D" clustering demonstration.  
5. **Calls the high-level "run_clustering()" function**, which performs:
   - preprocessing,  
   - clustering,  
   - metric computation,  
   - plotting.  
6. **Saves results** in the "demo_output/" directory, including:
   - a "2D" cluster plot,  
   - an elbow plot,  
   - a CSV file containing the clustered dataset.  

The demo script provides a complete example of how the package is intended
to be used in practice.


# Summary

"cluster_maker" offers a full pipeline for clustering analysis, including:

- synthetic data generation,  
- exploratory statistics,  
- preprocessing utilities,  
- both manual and scikit-learn clustering algorithms,  
- evaluation metrics,  
- visualisation tools,  
- a high-level orchestration interface,  
- and a ready-to-use demo script.

Its modular structure allows students to understand each step of the workflow,
debug individual components, and extend the package with new methods or analyses.

This makes "cluster_maker" an ideal educational tool for learning Python,
data processing, and clustering techniques in a clear and structured way.
