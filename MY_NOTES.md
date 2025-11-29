# cluster_maker

Cluster maker is the PACKAGE, its a folder (cluster_maker) containing python modules

Each .py file inside it are modules with specific responsibilities.
This is a typical modular architecture: each file contains a set of related functions, and the package combines them together.

Module	                        What it does
dataframe_builder.py	        Build empty dataframe schema + simulate synthetic datasets
data_analyser.py	            Summary stats, correlations, describing data
data_exporter.py	            Save CSVs, formatted tables, summaries
preprocessing.py	            Feature selection, scaling
algorithms.py	                Your K-means implementation, centroid init, updating, etc
evaluation.py	                Inertia, silhouette, elbow curve calculations
plotting_clustered.py	        Plot clusters and elbow curve
stability.py (created by me)	Cluster stability score
interface.py	                High-level function run_clustering() that strings everything together

# __init__.py file

Now the __init__.py file: This is what makes Python treat the folder as a package

**It controls what is accessible when the user writes import cluster_maker**
__init__.py exposes a clean public API (Application programming interface)
A public API is the set of fucntions and tools from a package that a user is meant to use.
The __init__.py file selectively presents only the important, user-facing functions of the package, making cluster_maker simple and pleasant to import and use.

**It defines __all__ = an official export list**
defines all functions that belong to the public API, helpful if a user does
from cluster_maker import *       or
dir(cluster_maker)

**It hides internal modules**
Anything not in __all__ is considered "private" even though it technically exists

Summary: 
- the init file turns the folder into a package
- imports seclected functions from each module
- makes them available directly under cluster_maker
- defines the public API wia __all__


# dataframe_builder.py file

This module contains two core funtions:
- define_dataframe_structure() : builds the dataframe that holds cluster centres
- simulate_data() : uses those cluster centres to generate synthetic clustered data with Gaussian noise

This file is responsible for the first stage of the clustering pipeline: 
**creating synthetic datasets** that later modules will analyse, preprocess, cluster, evaluate and plot.





