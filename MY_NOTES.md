
# cluster_maker

cluster_maker is the PACKAGE, its a folder containing python modules

Each .py file inside it are the modules with specific responsibilities. This is a typical modular architecture: each file contains a set of related functions, and the package combines them together.

| Module                          | What it does                                                     |
|---------------------------------|------------------------------------------------------------------|
| `dataframe_builder.py`          | Build empty dataframe schema + simulate synthetic datasets       |
| `data_analyser.py`              | Summary stats, correlations, describing data                     |
| `data_exporter.py`              | Save CSVs, formatted tables, summaries                           |
| `preprocessing.py`              | Feature selection, scaling                                       |
| `algorithms.py`                 | Your K-means implementation, centroid init, updating, etc        |
| `evaluation.py`                 | Inertia, silhouette, elbow curve calculations                    |
| `plotting_clustered.py`         | Plot clusters and elbow curve                                    |
| `stability.py` *(created by me)*| Cluster stability score                                          |
| `interface.py`                  | High-level function `run_clustering()` that strings everything together |


## `__init__`.py file

Now the `__init__`.py file: This is what makes Python treat the folder as a package

**It controls what is accessible when the user writes import cluster_maker**

`__init__`.py exposes a clean public API (Application programming interface). 
A public API is the set of fucntions and tools from a package that a user is meant to use. 
The `__init__`.py file selectively presents only the important, user-facing functions of the package,  making cluster_maker simple and pleasant to import and use.

**It defines `__all__` an official export list**

It defines all functions that belong to the public API, helpful if a user does
from cluster_maker import *       or
dir(cluster_maker)

**It hides internal modules**

Anything not in `__all__` is considered "private" even though it technically exists

------------------------------------------

Summary: 
- the init file turns the folder into a package
- imports seclected functions from each module
- makes them available directly under cluster_maker
- defines the public API via `__all__`


## dataframe_builder.py file

This module contains two functions:
- **define_dataframe_structure():** builds the dataframe that holds cluster centres
- **simulate_data():** uses those cluster centres to generate synthetic clustered data with Gaussian noise

This file is responsible for the first stage of the clustering pipeline: 
**creating synthetic datasets** that later modules will analyse, preprocess, cluster, evaluate and plot.

------------------------------------------

**define_dataframe_structure():** turns user specifications into a clean dataframe of cluster centres
- INPUT:
    - a list of dictionaries (column_specs), of the form key=String, Value=Any. Each dictionary describes a feature/column and its cluster center values
- OUTPUT:
    - a pandas DataFrame
- Extracts 'reps' from each spec and collects their length, checks that all features have the same number of cluster centre values, by reducing a list into a set and checking the length of the set is 1
- defines n_clusters as the number of 'reps'
- creates an empty dictionary called data
- iterates through each column/feature, retrieving the column name, and the reps, adn adding to the dictionary data, whilst handling various errors
- coverts the dictionary data into a pandas DataFrame called seed_df
- labels the index 'cluster_id' for clarity
- HANDLES ERRORS:
    - empty input list (ValueError)
    - incorrect 'reps' length for specific features (ValueError)
    - missing 'name' or 'reps' keys (ValueError)
    - 'reps' not being a sequence (TypeError)
    - re-check that reps' lists are the same length (ValueError)

------------------------------------------

**simulate_data():** generates synthetic clustered data by adding Gaussian noise around given cluster centres in seed_df
- INPUT:
    - seed_df (`pd.DataFrame`) Rows = cluster centres, columns = features
    - n_points (`int`, default `100`) Total number of points to simulate
    - cluster_std (`float`, default `1.0`) Standard deviation of noise around each cluster centre
    - random_state (`int | None`) Seed for reproducibility
- OUTPUT:
    - data (`pd.DataFrame`) A DataFrame with: the simulated feature values and a `"true_cluster"` column indicating the cluster each point came from
- sets random seed, converts seed_df into numpy array
- defines n_clusters and n_features
- distributes points as evenly as possible across clusters
- pairs each centre with it't assigned number of points and iterates through this
- generates gaussian noise and adds noise to the centre to create points (using broadcasting)
- saves features into a dictionary for each point, then puts all dicts into a list
- makes a DataFrame from the list of dictionaries
- HANDLES ERRORS:
    - if n_points, the number of data points to simulate is positive (ValueError)
    - that the cluster std is positive (ValueError)

## data_analyser.py file

This module contains three functions:
- **calculate_descriptive_statistics():** returns the count, mean, std, min/max and quartiles for the data
- **calculate_correlation():** computes the correlation matrix for numeric columns
- **summarise_numeric_columns():** creates a custom summary table with mean, std, mix/max and number of missing values


This file provides basic exploratory data analysis utilities for the cluster_maker package. Its functions help users understand the dataset before clustering, by generating descriptive statistics, correlations, and simple summaries of numeric variables.

------------------------------------------


**calculate_descriptive_statistics():** creates a DataFrame describing the standard descriptive statistics for each numeric column 
- INPUT:
    - Pandas DataFrame of data
- OUTPUT:
    - Pandas DataFrame of summary statistics
- The function checks that the input is a DataFrame, and if it is, it returns the standard descriptive statistics for each numeric column (count, mean, standard deviation, quartiles, and min/max). 
- It’s essentially a safe wrapper around data.describe() with type checking and documentation.
- HANDLES ERRORS:
-   if input data is not in a Pandas DataFrame (TypeError)

------------------------------------------

**calculate_correlation():** creates a DataFrame that returns the correlation matrix for all numeric columns
- INPUT:
    - a Pandas DataFrame of data
- OUTPUT:
    - a Pandas DataFrame of correlations
- The function checks that the input is a pandas DataFrame, and then computes and returns the correlation matrix for all numeric columns. 
- It serves as a safe and clean wrapper around data.corr() with type checking to ensure valid input.
- HANDLES ERRORS:
    - if the input isn't a Pandas DataFrame

------------------------------------------

**summarise_numeric_columns():** provides a clean, human-readable summary of all numeric columns within a DataFrame
- INPUT:
    - DataFrame of data
- OUTPUT:
    - DataFrame of summary statistics of numeric columns
- It automatically identifies numeric features
- Non-numeric columns are safely ignored, with a gentle warning to the user
- computes key descriptive statistics (mean, standard deviation, minimum, maximum, and the number of missing values)
- organises these results into a dictionary then into tidy summary table (DataFrame)
- HANDLES ERRORS:
    - if the input isn't a Pandas DataFrame

## data_exporter.py file

This module contains 3 functions:
- **export_to_csv():** Exports a pandas DataFrame to a CSV file with optional custom delimiter and index inclusion
- **export_formatted():** Exports a pandas DataFrame as a neatly formatted plain-text table, either to a file or to an open file-like object
- **export_summary():** Exports a numeric summary DataFrame to both a CSV file and a human-readable formatted text file

This file handles exporting data and summaries to external files.
It allows users to save dataframes as CSV files or formatted text tables, supporting reproducible workflows and easy sharing of outputs.

------------------------------------------

**export_to_csv():** Exports a pandas DataFrame to a CSV file with optional custom delimiter and index inclusion
- INPUT:
    - data (`pandas.DataFrame`) — the table of data to be exported.  
    - filename (`str`) — the name or path of the output CSV file.  
    - delimiter (`str`, default `","`) — the column separator to use in the CSV.  
    - include_index (`bool`, default `False`) — whether to write the DataFrame’s index to the file.

- OUTPUT:
    - None — the function performs a file-write action, saves a CSV to the cwd and returns nothing
- wrapper around Pandas .to_csv() function
- HANDLES ERRORS:
    - if data isnt a pandas dataframe (TypeError)

------------------------------------------

**export_formatted():** Exports a pandas DataFrame as a neatly formatted plain-text table, either to a file or to an open file-like object

- INPUT:
    - data (`pandas.DataFrame`) — the DataFrame to be exported.  
    - file (`str` or file-like object) — either a filename or an already opened file handle to write to.  
    - include_index (`bool`, default `False`) — whether the DataFrame’s index should be included in the formatted text output.
- OUTPUT:
    - None — the function writes writes a text file containing a formatted table but does not return a value.
- Converts the DataFrame into a formatted plain-text table
- If file is a filename (string), the function opens that file in write mode and writes the formatted table to it.
- If file is a file-like object (e.g., an already opened file, a StringIO, or a stream), it simply writes the text to the existing handle.
- HANDLES ERRORS:
    - if data is not a pandas dataframe (TypeError)

------------------------------------------

**export_summary():** Exports a numeric summary DataFrame to both a CSV file and a human-readable formatted text file
- INPUT:
    - summary_df (`pd.DataFrame`) Output of `summarise_numeric_columns()`, containing per-column statistics  
    - csv_path (`str`) File path for the CSV output  
    - txt_path (`str`) File path for the formatted text summary  
- OUTPUT:
    - None (no return)
    - Saves a **CSV file** containing the raw summary table  
    - Saves a **plain-text (.txt)** file containing a neatly formatted, readable report  
- Saves the DataFrame as a CSV
- Creates and opens a .txt file with UTF-8 encoding and writes a header.
- Write each row’s stats in a friendly format
- HANDLES ERRORS:
    - if summary_df is not a pandas DataFrame (TypeError)
    - if CSV saving fails (ValueError)
    - if Text file saving fails (ValueError)

## preprocessing.py file

This module contains 2 functions:
- **select_features():** Selects a subset of columns from a DataFrame and ensures that all chosen features are numeric
- **standardise_features():** Standardises all features to zero mean and unit variance using scikit-learn’s `StandardScaler`

This file performs data preprocessing before clustering.
It selects valid feature columns, checks they are numeric, and standardises features so clustering algorithms behave correctly.

------------------------------------------

**select_features():** Selects a subset of columns from a DataFrame and ensures that all chosen features are numeric
- INPUT:
    - data (`pd.DataFrame`) The original dataset
    - feature_cols (`List[str]`) A list of column names the user wants to use as features for clustering
- OUTPUT:
    - X_df (`pd.DataFrame`) A new DataFrame containing only the selected columns, guaranteed to be numeric
- Extracts only the columns the user specified, using .copy() avoids modifying the original DataFrame
- Identifies any selected columns that are not numeric (e.g., strings, categories, booleans, timestamps)
- *DOESN'T* ACTUALLY REMOVE THESE FROM X_df - potential fix????
- HANDLES ERRORS:
    - if any of the columns requested are missing (KeyError)
    - if any of the columns requested are non-numeric (TypeError), k-means clustering requires numeric input

------------------------------------------

**standardise_features():** Standardises all features to zero mean and unit variance using scikit-learn’s `StandardScaler`
- INPUT:
    - X (`np.ndarray`) A 2D NumPy array of shape `(n_samples, n_features)` containing the feature values.
- OUTPUT:
    - X_scaled** (`np.ndarray`) A NumPy array of the same shape, with each feature standardised
- It creates a StandardScaler object, which is a tool from scikit-learn that learns how to scale your data properly. It figures out the mean and std for each feature
- fit_transform computes the mean and std for each feature such that mean = 0, standard deviation = 1
- returns the standardised NumPy array
- HANDLES ERRORS:
    - if input isn't a NumPy array (TypeError)


## algorithms.py file

This module contains 5 functions:

- **init_centroids():** Randomly selects k data points from `X` (without replacement) to use as the initial centroids for k-means clustering
- **assign_clusters():** Computes the distance from each data point to each centroid and assigns each point to the closest centroid
- **update_centroids():** Recomputes each centroid by taking the mean of all points assigned to that cluster, and if a cluster has no points, randomly reinitialises its centroid
- **kmeans():** Runs a complete manual K-means clustering algorithm: initialises centroids → assigns points → updates centroids → repeats until convergence → returns final labels and centroids
- **sklearn_kmeans():** Runs scikit-learn’s built-in KMeans algorithm and returns the resulting cluster labels and centroids


This file implements the clustering algorithms used in the package.
It includes a manual K-Means implementation (with centroid initialisation, assignment, updating) and a wrapper for scikit-learn’s KMeans.

------------------------------------------

**init_centroids():** Randomly selects k data points from `X` (without replacement) to use as the initial centroids for k-means clustering
- INPUT:
    - X (`np.ndarray`) A 2D array of shape `(n_samples, n_features)` containing your dataset
    - k (`int`) The number of clusters (and therefore the number of centroids to pick)
    - random_state (`int | None`, optional) Seed for reproducibility; ensures the same centroids are chosen every run
- OUTPUT:
    - centroids (`np.ndarray`) A NumPy array containing the `k` selected initial centroids (shape `(k, n_features)`)
- Gets the number of rows in X (i.e., how many data points you have) this is n_samples
- choose k numbers from 1 to n_samples (number of data points), without replacement
- return only those rows from X, thats your centroids
- HANDLES ERRORS:
    - if k is not a positive integer (ValueError)
    - if k > number of data points (ValueError)

------------------------------------------

**assign_clusters():** Computes the distance from each data point to each centroid and assigns each point to the closest centroid
- INPUT:
    - X (`np.ndarray`) Shape `(n_samples, n_features)` — the dataset
    - centroids (`np.ndarray`) Shape `(k, n_features)` — current centroid positions.
- OUTPUT:
    - labels (`np.ndarray`) Shape `(n_samples,)` — for each sample, the index (0…k−1) of the nearest centroid
- For every sample and every centroid, compute (sample − centroid)
- Takes the L2 norm across the feature dimension
- Converts the (n_samples, k, n_features) tensor into (n_samples, k)
- Pick the closest centroid for each sample
- return vector of labels, length n_samples
- HANDLES ERRORS:
    - *DOESN'T* check inputs are numeric arrays
    - *DOESN'T* check that X and centroids have numeric shape

------------------------------------------

**update_centroids():** Recomputes each centroid by taking the mean of all points assigned to that cluster, and if a cluster has no points, randomly reinitialises its centroid
- INPUT:
    - X (`np.ndarray`) Shape `(n_samples, n_features)` — the dataset.
- labels (`np.ndarray`) Shape `(n_samples,)` — cluster assignments for each sample.
- k (`int`) Number of clusters.
- random_state (`int | None`, optional) Seed for reproducible handling of empty clusters
- OUTPUT:
    - new_centroids (`np.ndarray`) Shape `(k, n_features)` — the updated centroid positions
- gets number of features
- prepares a zeros array for new centroids
- Creates random generator for empty-cluster handling
- Loop over each cluster
- Identify samples belonging to this cluster using a boolean array selecting points assigned to cluster_id
- If empty cluster: re-initialise randomly
- Takes the average of all samples assigned to the cluster, sets that as the updated centroid position
- HANDLES ERRORS:
    - *Doesn't* check that X and labels have the same lengths (ValueError)
    - *Doesn't* check that labels contains values not in 0 - k-1 (IndexError)
    - *Doesn't* check that X contains non-numeric data (TypeError)

------------------------------------------

**kmeans():** Runs a complete manual K-means clustering algorithm: initialises centroids → assigns points → updates centroids → repeats until convergence → returns final labels and centroids
- INPUT:
    - X (`np.ndarray`) The dataset, shape `(n_samples, n_features)`
    - k (`int`) Number of clusters
    - max_iter (`int`, default `300`) Maximum number of K-means iterations
    - tol (`float`, default `1e-4`) Minimum centroid movement required to continue
    - random_state (`int | None`) Seed for reproducible centroid initialisation and empty-cluster handling
- OUTPUT:
    - Returns a tuple:
        1. **labels** (`np.ndarray`) — cluster assignment for each sample  
        2. **centroids** (`np.ndarray`) — final centroid coordinates (shape `(k, n_features)`)
- initialize centroids using init_centroids created earlier
- assign clusters using assign_clusters created earlier
- update_centroids using funtion created earlier
- Checking for convergence: calculate distance from new centroids to old centroids
- replace old centroids with new centroids
- if the distance from old to new centroids is small, break for loop
- Ensures labels match the final centroids
- HANDLES ERRORS:
    - if X is not a NumPy array (TypeError)

------------------------------------------

**sklearn_kmeans():** Runs scikit-learn’s built-in KMeans algorithm and returns the resulting cluster labels and centroids
- INPUT:
    - X (`np.ndarray`) Dataset of shape `(n_samples, n_features)`
    - k (`int`) Number of clusters for KMeans
    - random_state (`int | None`) Seed for reproducible clustering
- OUTPUT:
    - Returns a tuple:
        1. labels** (`np.ndarray`) Cluster assignment for each data point  
        2. Centroids (`np.ndarray`) Final centroid positions (shape `(k, n_features)`)
- Creates the scikit-learn KMeans model
    - n_init=10: runs KMeans 10 times with different centroid seeds internally (scikit-learn picks the best result)
- Fit model to data; Performs the entire K-means procedure
- Extract labels and centroids
- HANDLES ERRORS:
    - if X is not a NumPy array (TypeError)

## evaluation.py file

This module contains 3 functions:
- **compute_inertia():** Calculates the total within-cluster sum of squared distances (inertia), which measures how tightly the data points cluster around their centroids
- **silhouette_score_sklearn():** Calculates the silhouette score (a measure of clustering quality) using scikit-learn’s built-in function
- **elbow_curve():** Runs K-means for multiple values of k and returns a dictionary mapping each k to its inertia, enabling the elbow method

This file provides evaluation metrics for cluster quality, including inertia (WCSS), silhouette score, and functions for computing elbow curves across multiple values of k.

------------------------------------------

**compute_inertia():** Calculates the total within-cluster sum of squared distances (inertia), which measures how tightly the data points cluster around their centroids

- INPUT:
    - X (`np.ndarray`) Data matrix of shape `(n_samples, n_features)`
    - labels (`np.ndarray`) Cluster assignments for each sample `(n_samples,)`
    - centroids (`np.ndarray`) Centroid coordinates `(k, n_features)`
- OUTPUT:
    - inertia (`float`) The sum of squared distances between each sample and its assigned centroid
- Compute distances between each point and its assigned centroid, has shape (n_samples, n_features)
- Square the distances and sum them
- Return inertia as a float
- HANDLES ERRORS:
    - if the numebr of samples in X doesnt match the number of labels (ValueError)

------------------------------------------

**silhouette_score_sklearn():** Calculates the silhouette score (a measure of clustering quality) using scikit-learn’s built-in function
- INPUT:
    - X (`np.ndarray`) The dataset, shape `(n_samples, n_features)`
    - labels (`np.ndarray`) Cluster assignment for each sample `(n_samples,)`
- OUTPUT:
    - score (`float`) The silhouette score, ranging from **−1 to +1**, where:
        - +1 = very well-separated clusters  
        - 0 = overlapping clusters  
        - −1 = misassigned points  
- Computes the silhouette score using scikit-learn and returns as a float
- HANDLES ERRORS:
    - if there are are less than 2 centroids (ValueError) 

------------------------------------------

**elbow_curve():** Runs K-means for multiple values of k and returns a dictionary mapping each k to its inertia, enabling the elbow method

- INPUT:
    - X (`np.ndarray`) Dataset of shape `(n_samples, n_features)`
    - k_values (`List[int]`) A list of k values to test, e.g. `[1, 2, 3, 4, 5]`
    - random_state (`int | None`) Seed for reproducibility
    - use_sklearn (`bool`, default `True`)  
        - If `True`, use scikit-learn’s KMeans  
        - If `False`, use the manual `kmeans()` implementation
- OUTPUT:
    - inertia_dict (`Dict[int, float]`) A mapping where: { k1: inertia_for_k1, k2: inertia_for_k2, ... }
- Prepares an empty dictionary that stores inertia results for each k
- Loops through each k value, find the cluster labels and centroid positions using sklearn, or manually depending on input
- Computes inertia for this value of k
- Stores the inertia value in the dictionary
- Returns the final mapping
- HANDLES ERRORS:
    - if k's are not positive integers (ValueError)

## plotting_clustered.py file

This module contains 2 functions:
- **plot_clusters_2d():** Creates a 2D scatter plot of clustered data using the first two features, with optional centroid markers
- **plot_elbow():** Creates an elbow plot by graphing inertia values against different k values, helping identify the optimal number of clusters

This file contains plotting utilities for visualising clustering results.
It generates 2D scatter plots with cluster labels and centroids, and plots inertia vs. k for the elbow method.

------------------------------------------

**plot_clusters_2d():** Creates a 2D scatter plot of clustered data using the first two features, with optional centroid markers
- INPUT:
    - X (`np.ndarray`) Dataset of shape `(n_samples, n_features)`  
    - labels (`np.ndarray`) Cluster assignment for each point `(n_samples,)`
    - centroids (`np.ndarray | None`) Optional centroid array `(k, n_features)`
    - title (`str | None`) Optional plot title
- OUTPUT:
    - Returns a tuple:
        - fig (`matplotlib.figure.Figure`)
        - ax (`matplotlib.axes.Axes`)
- Plot the samples coloured by cluster label
- Optionally plot centroids
- Add labels, optional title, add colour bar
- HANDLES ERRORS:
    - if X has less than 2 features (ValueError)


------------------------------------------

**plot_elbow():** Creates an elbow plot by graphing inertia values against different k values, helping identify the optimal number of clusters
- INPUT:
    - k_values (`List[int]`) A list of k values (e.g., `[1,2,3,4,5]`)
    - inertias (`List[float]`) Corresponding inertia values computed for each k
    - title (`str`, default `"Elbow Curve"`) Title of the plot
- OUTPUT: (as a tuple)
    - fig (`matplotlib.figure.Figure`)  
    - ax (`matplotlib.axes.Axes`)  
- plots the elbow curve, k on x-axis, inertias on y-axis
- adds axis labels, titles, grid
- HANDLES ERRORS:
    - if number of k and number of inertias are different (ValueError)


## stability.py file

This module contains 1 function:
- **cluster_stability_score()**

This file implements the cluster stability diagnostic.
It measures how consistent clustering results are across repeated noisy perturbations of the dataset, correcting for label switching to produce a stability score between 0 and 1.

------------------------------------------

**cluster_stability_score():** Measures how stable clustering results are by repeatedly adding small noise to the data, re-running KMeans, fixing label switching, and computing how often each pair of points ends up in the same cluster

- INPUT:
    - X (`np.ndarray`) Dataset of shape `(n_samples, n_features)`
    - k (`int`) Number of clusters
    - n_runs (`int`, default `20`) Number of clustering runs (1 base run + 19 noisy runs)
    - noise_scale (`float`, default `0.05`) Strength of added Gaussian noise before each additional clustering run
    - random_state (`int | None`) Seed for reproducibility
- OUTPUT:
    - stability (`float`) A value in **[0, 1]**:
        - `1` = perfectly stable clustering  
        - `0` = totally unstable clustering 
- Chooses which clustering algorithm to use (kmeans or sklearn_kmeans) based on the algorithm argument
- Sets up a co-occurrence matrix to record how often each pair of points ends up in the same cluster
- Runs an initial “base” clustering on the original dataset.
- Sorts the base centroids and remaps cluster labels so that label ordering is consistent across all runs (fixes label switching).
- Updates the co-occurrence matrix using the aligned base labels.
- Repeats clustering on noisy versions of the dataset for n_runs - 1 additional runs.
- For each noisy run:
    - Generates Gaussian noise and adds it to the dataset.
    - Clusters the noisy data using the chosen algorithm.
- Sorts centroids again and aligns labels for consistency.
- Updates the co-occurrence matrix based on which points share the same aligned label.
- Averages the co-occurrence counts across all runs to form the stability matrix.
- Computes the final stability score as the mean of all off-diagonal entries in the stability matrix.
- Returns a float in [0, 1], where higher values indicate more stable, robust clustering.
- HANDLES ERRORS:
    - if X is not a NumPy array (TypeError)
    - if X has fewer than 2 samples (ValueError)
    - if k is not a positive integer (ValueError)
    - if k > number of samples (ValueError)
    - if n_runs < 1 (ValueError)
    - if noise_scale is negative (ValueError)
    - if algorithm is not "kmeans" or "sklearn_kmeans" (ValueError)
    - if base clustering fails (RuntimeError)
    - if clustering fails during any noisy run (RuntimeError)


## interface.py file

This module contains 1 function:
- **run_clustering()**

This file provides the high-level run_clustering interface, which ties together the entire pipeline:
loading data → preprocessing → clustering → evaluation → plotting → optional exporting.
It acts as the “one-call” function for running the full workflow.

------------------------------------------

**run_clustering():** Runs the entire clustering pipeline end-to-end:  
loading data → selecting features → standardising → clustering → evaluation → plotting → optional export → returns all results in a dictionary.
- INPUT:
    - input_path (`str`) Path to the input CSV file
    - feature_cols (`List[str]`) Columns to use as features
    - algorithm (`"kmeans"` or `"sklearn_kmeans"`, default `"kmeans"`) Which clustering implementation to use
    - k (`int`, default `3`) Number of clusters
    - standardise (`bool`, default `True`) Whether to standardise features
    - output_path (`str | None`, default `None`) If provided, save the labelled data to CSV
    - random_state (`int | None`) Seed for reproducibility
    - compute_elbow (`bool`, default `False`) Whether to compute elbow curve metrics
    - elbow_k_values (`List[int] | None`) List of k values for elbow method (autogenerated if None)
    - compute_stability (`bool`, default `False`) NEW FEATURE — whether to compute cluster stability
- OUTPUT: (returned in a dict)
    - `"data"` — DataFrame with a new `"cluster"` column  
    - `"labels"` — final cluster labels (NumPy array)  
    - `"centroids"` — final centroid positions  
    - `"metrics"` — dict containing:
    - `"inertia"`
    - `"silhouette"`
    - `"stability"` (if computed)
    - `"fig_cluster"` — 2D cluster plot (matplotlib Figure)
    - `"fig_elbow"` — elbow plot (or `None`)
    - `"elbow_inertias"` — dict mapping k → inertia
    - `"stability"` — stability score (if computed)
- Reads a CSV file into a pandas DataFrame
- Converts selected features to a NumPy array
- If standardise=True, scales features to mean 0 and std 1 using standardise_features()
- Run the clustering algorithm depending on chosen algorithm
- Compute evaluation metrics - Inerita, Silhouette score (if feasible), stability score (my new feature)
- Add cluster labels to the original DataFrame
- export labelled data to CSV
- Generate the cluster plot
- Optional: compute elbow curve
- Return everything in one dictionary
- HANDLES ERRORS:
    - if algorithm is not kmeans or sklearn_kmeans (ValueError)
    - if silhouette score returns an error, it outputs none in the dictonary


#### NOTES ON CLUSTERING EVALUATION METHODS


--------------------------------

1. INERTIA (Within-Cluster Sum of Squares)
------------------------------------------
Definition:
    Inertia measures how internally coherent clusters are.
    It is the sum of squared distances between each data point
    and the centroid of the cluster it belongs to.

Formula:
    Inertia = Σ (for j = 1 to k) Σ (xᵢ ∈ Cⱼ) || xᵢ - μⱼ ||²
        where:
            Cⱼ = set of points in cluster j
            μⱼ = centroid of cluster j

Interpretation:
    - Lower inertia → more compact clusters
    - Higher inertia → clusters are more spread out
    - Inertia always decreases as k increases, so it cannot be used alone
      to find the optimal number of clusters.

--------------------------------------------------------------

2. THE ELBOW METHOD
-------------------
Purpose:
    Used to find an appropriate number of clusters (k) by analyzing
    how inertia changes as k increases.

Procedure:
    1. Run K-Means for a range of k values (e.g., 1–10).
    2. Compute inertia for each k.
    3. Plot k (x-axis) vs. inertia (y-axis).
    4. Identify the "elbow" — the point where inertia stops
       decreasing rapidly. This indicates the best trade-off
       between compactness and simplicity.

Interpretation:
    - Before the elbow: adding clusters greatly reduces inertia.
    - After the elbow: adding clusters yields minimal improvement.

--------------------------------------------------------------

3. SILHOUETTE SCORE
-------------------
Definition:
    The silhouette score measures how well-separated and compact
    the clusters are. It assesses the quality of the clustering
    structure.

For each sample i:
    a(i) = mean intra-cluster distance (to points in the same cluster)
    b(i) = mean nearest-cluster distance (to points in the closest other cluster)

    s(i) = (b(i) - a(i)) / max(a(i), b(i))

Range:
    -1 ≤ s(i) ≤ 1
        - s(i) ≈ 1 → well-clustered point (far from other clusters)
        - s(i) ≈ 0 → on the boundary between clusters
        - s(i) < 0 → may be misclassified

Overall silhouette score:
    Average of all s(i) values.

Usage:
    - Compute for different k values.
    - Choose k with the highest average silhouette score
      (best balance of separation and cohesion).

--------------------------------------------------------------

Summary:
    - Inertia: measures compactness (used in Elbow Method)
    - Elbow Method: visual heuristic to choose k
    - Silhouette Score: quantitative measure of cluster quality
"""
