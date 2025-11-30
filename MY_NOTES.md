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

This module contains two core functions:
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

This module contains three core functions:
- **calculate_descriptive_statistics():** returns the count, mean, std, min/max and quartiles for the data
- **calculate_correlation():** computes the correlation matrix for numeric columns
- **summarise_numeric_columns():** creates a custom summary table with mean, std, mix/max and number of missing values


This file provides basic exploratory data analysis utilities for the cluster_maker package. Its functions help users understand the dataset before clustering, by generating descriptive statistics, correlations, and simple summaries of numeric variables.


**calculate_descriptive_statistics():**
- INPUT:
    - Pandas DataFrame of data
- OUTPUT:
    - Pandas DataFrame of summary statistics
- The function checks that the input is a DataFrame, and if it is, it returns the standard descriptive statistics for each numeric column (count, mean, standard deviation, quartiles, and min/max). 
- It’s essentially a safe wrapper around data.describe() with type checking and documentation.

Handles Errors:
- if input data is not in a Pandas DataFrame (TypeError)


**calculate_correlation():**
- INPUT:
    - a Pandas DataFrame of data
- OUTPUT:
    - a Pandas DataFrame of correlations
- The function checks that the input is a pandas DataFrame, and then computes and returns the correlation matrix for all numeric columns. 
- It serves as a safe and clean wrapper around data.corr() with type checking to ensure valid input.

Handles Errors:
- if the input isn't a Pandas DataFrame



**summarise_numeric_columns():**
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