# cluster_maker Package — Description and Module Overview

The `cluster_maker` package is a small, modular Python library designed to demonstrate and support teaching of clustering workflows in data science. Its purpose is to show every stage of a full clustering pipeline in a clear and accessible way, beginning with data preparation and ending with visualisation and export of results. The system is organised into several separate modules, each handling one specific part of the workflow, and these modules combine to form a complete clustering process when used together.

## Package Structure

The package consists of the following files: `__init__.py`, `algorithms.py`, `data_analyser.py`, `data_exporter.py`, `dataframe_builder.py`, `evaluation.py`, `interface.py`, `plotting_clustered.py`, and `preprocessing.py`. Each file contributes a distinct component of the clustering workflow. The overall design encourages modularity and teaches how real-world projects separate functionality into focused parts.

## Module Descriptions

The `__init__.py` file sets up the package and exposes the main public functions so that users do not need to import individual modules manually. This allows simple import statements such as `from cluster_maker import run_clustering`.

The `algorithms.py` module contains the actual clustering algorithms. It includes a simple custom implementation of the K-Means algorithm which assigns points to centroids and updates them iteratively, and it also contains a wrapper around scikit-learn’s KMeans implementation. These two approaches allow both educational exploration and comparison against a standard library version.

The `data_analyser.py` module provides utility functions for examining data. Its purpose is to support inspection, summarisation, and optional analysis of datasets before or after clustering. This module is useful for exploring distributions or understanding properties of the data that has been clustered. In task 3, a function was added that calculates the column statistics for numerical columns in a df.

The `data_exporter.py` module handles the exporting of results. It contains a function that writes pandas DataFrames, including cluster labels if present, to a CSV file. This allows the clustered dataset to be saved and used outside the program. In task 3, we also created a function that saves the summary table from data_analyser.py as a csv and then presents it to the user in a readable text file.

The `dataframe_builder.py` module is responsible for constructing artificial clustered datasets. It includes functionality to define a table of cluster centres and then simulate noisy data around those centres. This is particularly helpful for testing algorithms or demonstrating how clustering behaves on manufactured datasets. The module returns simulated points and also provides the true cluster label for each generated point.

The `evaluation.py` module implements the metrics required to assess clustering quality. It calculates inertia, which measures the compactness of clusters, and the silhouette score, which evaluates how well each point fits within its assigned cluster relative to others. It also supports the elbow method by computing inertia for a range of cluster counts, which helps determine an appropriate value of k.

The `interface.py` module acts as the high-level controller for the entire package. It contains the `run_clustering` function, which orchestrates the full workflow. This function loads the input CSV file, selects and validates features, optionally standardises them, runs the chosen clustering algorithm, computes evaluation metrics, generates plots, and saves the final labelled dataset. It returns a dictionary containing all results, including figures and evaluation measurements. This file represents the central engine that binds all other modules together.

The `plotting_clustered.py` module generates visualisations. It creates a 2D scatter plot of clustered data, marking cluster centroids, and also produces an elbow plot when requested. These plots are returned as matplotlib figure objects and are saved by the demo script, making the results easy to inspect.

The `preprocessing.py` module handles preparation of the data prior to clustering. It verifies that selected features exist and are numeric, returning a clean DataFrame of those features. It also standardises numerical values so that all features have comparable scales, which improves clustering performance and prevents features with larger numeric ranges from dominating the algorithm.

## How the Modules Work Together

A typical use of the package begins with the loading of a CSV file through the `run_clustering` function in `interface.py`. The chosen feature columns are passed through the preprocessing module to ensure they are valid and numerical, and they may be standardised. The processed data is then clustered using the algorithms provided in `algorithms.py`. After clustering, the evaluation module measures the quality of the result, and the plotting module produces visual output that helps interpret the clustering behaviour. Finally, the data exporter saves the labelled dataset, and additional analytical functions may be applied through `data_analyser.py`.

## Conclusion

Overall, the `cluster_maker` package provides a fully-worked example of a complete clustering system. Each module is focused on a clear individual responsibility, and together they form a coherent pipeline that is ideal for teaching and experimentation. The design encourages good programming practices such as modularity, clarity, and separation of concerns while demonstrating the full lifecycle of a clustering analysis workflow.
