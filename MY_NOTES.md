# cluster_maker

Cluster maker is the PACKAGE, its a folder (cluster_maker) containing python modules

Each .py file inside it are modules with specific responsibilities.
This is a typical modular architecture: each file contains a set of related functions, and the package combines them together.

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

defines all functions that belong to the public API, helpful if a user does
from cluster_maker import *       or
dir(cluster_maker)

**It hides internal modules**
Anything not in `__all__` is considered "private" even though it technically exists

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


Handles Errors: 
- empty input list (ValueError)
- incorrect 'reps' length for specific features (ValueError)
- missing 'name' or 'reps' keys (ValueError)
- 'reps' not being a sequence (TypeError)
- re-check that reps' lists are the same length (ValueError)


**simulate_data():** generates synthetic clustered data based on the cluster centres in seed_df
- INPUT:
    - 
- OUTPUT:
    - 
- 
- 
- 
- 

Handles Errors:
- 
- 
- 

## data_analyser.py file

This module contains three functions:
- **calculate_descriptive_statistics():** returns the count, mean, std, min/max and quartiles for the data
- **calculate_correlation():** computes the correlation matrix for numeric columns
- **summarise_numeric_columns():** creates a custom summary table with mean, std, mix/max and number of missing values


This file provides basic exploratory data analysis utilities for the cluster_maker package. Its functions help users understand the dataset before clustering, by generating descriptive statistics, correlations, and simple summaries of numeric variables.


**calculate_descriptive_statistics():** creates a DataFrame describing the standard descriptive statistics for each numeric column 
- INPUT:
    - Pandas DataFrame of data
- OUTPUT:
    - Pandas DataFrame of summary statistics
- The function checks that the input is a DataFrame, and if it is, it returns the standard descriptive statistics for each numeric column (count, mean, standard deviation, quartiles, and min/max). 
- It’s essentially a safe wrapper around data.describe() with type checking and documentation.

Handles Errors:
- if input data is not in a Pandas DataFrame (TypeError)


**calculate_correlation():** creates a DataFrame that returns the correlation matrix for all numeric columns
- INPUT:
    - a Pandas DataFrame of data
- OUTPUT:
    - a Pandas DataFrame of correlations
- The function checks that the input is a pandas DataFrame, and then computes and returns the correlation matrix for all numeric columns. 
- It serves as a safe and clean wrapper around data.corr() with type checking to ensure valid input.

Handles Errors:
- if the input isn't a Pandas DataFrame



**summarise_numeric_columns():** provides a clean, human-readable summary of all numeric columns within a DataFrame
- INPUT:
    - DataFrame of data
- OUTPUT:
    - DataFrame of summary statistics of numeric columns
- It automatically identifies numeric features
- Non-numeric columns are safely ignored, with a gentle warning to the user
- computes key descriptive statistics (mean, standard deviation, minimum, maximum, and the number of missing values)
- organises these results into a dictionary then into tidy summary table (DataFrame)


Handles Errors:
- if the input isn't a Pandas DataFrame

## data_exporter.py file

This module contains 3 functions:
- **export_to_csv():** Exports a pandas DataFrame to a CSV file with optional custom delimiter and index inclusion
- **export_formatted():** Exports a pandas DataFrame as a neatly formatted plain-text table, either to a file or to an open file-like object
- **export_summary():**

**export_to_csv():** Exports a pandas DataFrame to a CSV file with optional custom delimiter and index inclusion
- INPUT:
    - data (`pandas.DataFrame`) — the table of data to be exported.  
    - filename (`str`) — the name or path of the output CSV file.  
    - delimiter (`str`, default `","`) — the column separator to use in the CSV.  
    - include_index (`bool`, default `False`) — whether to write the DataFrame’s index to the file.

- OUTPUT:
    - None — the function performs a file-write action, saves a CSV to the cwd and returns nothing
- wrapper around Pandas .to_csv() function

Handles Errors:
- if data isnt a pandas dataframe (TypeError)

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

Handles Errors:
- if data is not a pandas dataframe

**export_summary():** Exports a numeric summary DataFrame to both a CSV file and a human-readable formatted text file
- INPUT:
    - summary_df (`pd.DataFrame`) Output of `summarise_numeric_columns()`, containing per-column statistics  
    - csv_path (`str`) File path for the CSV output  
    - txt_path (`str`) File path for the formatted text summary  
- OUTPUT:
    - 
- 
- 
- 
- 

Handles Errors:
- 
- 
- 

## preprocessing.py file

This module contains 2 functions:
- **select_features():**
- **standardise_features():**


## algorithms.py file

This module contains 5 functions:
- **init_centroids():**
- **assign_clusters():**
- **update_centroids():**
- **kmeans():**
- **sklearn_kmeans():**


## evaluation.py file


## plotting_clustered.py file


## stability.py file


## interface.py file


## CLUSTERING EVALUATION METHODS
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
