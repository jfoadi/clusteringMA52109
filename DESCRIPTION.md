# DESCRIPTION.md
# cluster_maker — Package Overview

## 1. Overview

cluster_maker is a Python package designed to support small-scale clustering analysis workflows for teaching and demonstration purposes.
It provides:

Creation of synthetic clustered datasets

Preprocessing and feature selection

Multiple clustering algorithms

Visualisation of clustering results

Evaluation metrics

High-level functions that orchestrate the entire workflow from raw CSV input to processed output

The package is modular, clear, and intended for non-expert users as well as students learning data science.

## 2. Package Structure
cluster_maker/
    __init__.py
    dataframe_builder.py
    preprocessing.py
    algorithms.py
    interface.py
    evaluation.py
    plotting_clustered.py
    data_analyser.py
    data_exporter.py


Each module has a distinct responsibility, contributing to a clean and readable system design.

## 3. Module-by-Module Description
### 3.1 dataframe_builder.py

Provides tools for creating seed dataframes representing cluster centres and simulating synthetic clustered datasets.

Key functions:

define_dataframe_structure(column_specs)
Builds a DataFrame where each column corresponds to a feature and each row corresponds to a cluster centre.

simulate_data(seed_df, n_points, cluster_std, random_state)
Generates synthetic data around the given centres, adding noise, and returning a labelled dataset with a "true_cluster" column.

Typical use case:
Testing clustering algorithms or demonstrating how cluster structure emerges from different centre configurations.

### 3.2 preprocessing.py

Contains data preprocessing utilities.

Likely features include:

Standardisation / normalisation

Handling missing values

Filtering or selecting subsets of columns

These operations prepare raw data before clustering.

### 3.3 algorithms.py

Implements the clustering algorithms available in the package.

Expected functionality:

KMeans clustering (default option)

Possibly extensions such as hierarchical clustering or others depending on the project

The functions here operate on preprocessed numeric data.

### 3.4 evaluation.py

Calculates metrics to assess the quality of clustering results.

Typical metrics:

Inertia / within-cluster sum of squares

Elbow score / diagnostic measures

Silhouette score (if included)

The results are returned to the interface for display or export.

### 3.5 plotting_clustered.py

Handles visualisation of clustered data.

Common outputs:

2D scatter plots showing clusters

Cluster centres

Elbow plot for choosing the number of clusters

Plots are returned as Matplotlib Figure objects so they can be saved by demo scripts or user code.

### 3.6 interface.py

The high-level orchestrator module.

This is the main user-facing API and coordinates:

Loading CSV data

Selecting numeric features

Preprocessing

Running the chosen clustering algorithm

Computing metrics

Generating plots

Exporting results

Key functions:

run_clustering(...) – The primary end-to-end pipeline.

select_features(df, feature_cols) – Input checking and validation for feature selection.

The demo scripts in the demo/ folder call this module directly.

### 3.7 data_analyser.py (added in Task 3)

Provides generic data-analysis utilities.

Key function (added for the mock exam):

summarise_numeric(df)

Accepts a pandas DataFrame

Extracts all numeric columns

Computes mean, std, min, max, number of missing values for each column

Returns a new summary DataFrame

Ignores or warns about non-numeric columns

Used to produce human-readable summaries for CSV data.

### 3.8 data_exporter.py (added in Task 3)

Handles exporting summaries and other outputs.

New functions include:

Writing the summary DataFrame (from data_analyser.py) to CSV

Creating a human-readable text file with one line per column

Error handling when directories or paths do not exist

Supports automated reporting of data characteristics.

## 4. Demo Scripts

Located in the demo/ directory.

cluster_analysis.py

Demonstrates a full clustering workflow using the package.
It:

Reads a CSV file from the command line

Prints an introductory explanation of what it will compute

Shows a sample of the dataset (df.head())

Chooses two numeric features automatically

Reminds the user of the model being applied (KMeans with k=3)

Runs clustering via run_clustering(...)

Saves:

Clustered dataset (CSV)

Cluster plot

Elbow plot (if computed)

Reports metrics clearly on screen

This script is designed for non-expert users and emphasises clarity, guidance, and robustness.

## 5. Overall Workflow

A typical use of the package follows these steps:

Load raw data (CSV)

Select numeric features

Preprocess / standardise

Apply clustering algorithm

Compute metrics to evaluate the clustering

Produce visualisations

Export results and summaries

Users can call run_clustering(...) directly or use the demo scripts for guided examples.

## 6. Intended Audience and Purpose

cluster_maker is designed primarily for:

Students learning clustering and data science

Teachers demonstrating clustering workflows

Anyone needing small, easy-to-understand clustering utilities

It emphasises:

clarity

modularity

good documentation

clean workflow design