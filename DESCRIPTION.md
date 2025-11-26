cluster_maker: An Educational Clustering Package

The cluster_maker package is designed to provide a comprehensive, yet simplified, workflow for simulating and analyzing clustered data, primarily for educational and practical assessment purposes.

Core Functionality Overview

The package breaks down the data science pipeline into modular components, allowing users to move from raw data to actionable insights and visualizations.

Data Management and Analysis (dataframe_builder.py, data_analyser.py)

These modules handle the input and initial understanding of the data. They allow for the simulation of clustered datasets (defining "seed" cluster centers) and compute basic statistical properties and correlation matrices for exploratory data analysis.

Preprocessing and Feature Engineering (preprocessing.py)

This component is crucial for preparing raw data for model training. It includes tools for feature selection and data standardization (scaling features to a common range, like Z-score).

Clustering Algorithms (algorithms.py)

The package offers two parallel approaches to clustering:

A simple, manual K-Means implementation for transparency and learning.

A robust wrapper for the production-grade scikit-learn KMeans algorithm.

Evaluation and Visualization (evaluation.py, plotting_clustered.py)

To assess the quality of the clustering solution, the package provides metrics such as Inertia (within-cluster sum of squares) and the Silhouette Score. It also generates data for and plots the Elbow Curve to aid in selecting the optimal number of clusters (K). Visualization tools include scatter plots of 2D clusters and the Elbow Curve plot.

High-Level Interface (interface.py)

The package is controlled by a high-level function, run_clustering, which orchestrates the entire workflow—from reading data and preprocessing to running the algorithm, evaluating, and plotting the results—with a single, user-friendly call.