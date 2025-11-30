###
## cluster_maker
## James Foadi - University of Bath
## November 2025
###

from __future__ import annotations
# this import improves type hinting by letting annotations refer to types as strings
# tells Python to store type hints as strings instead of evaluating them immediately. 
# This avoids name errors, supports forward references, improves performance, 
# and makes type hints more flexible.

from typing import List, Dict, Any, Sequence
# this import is used for type hinting, makes the function signatures clearer and more strict
# typing is a module in the Python standard library that provides type-hinting tools to describe 
# the expected types of variables, function arguments, and return values.


import numpy as np
# will be used for random number generation
import pandas as pd
# will be used to build the seed dataframe and the final output


# this is the function that turns user specifications into a clean dataframe of cluster centres
def define_dataframe_structure(column_specs: List[Dict[str, Any]]) -> pd.DataFrame:
    # A type hint is optional metadata added to Python code that specifies the expected data types 
    # of variables, function parameters, and return values. It improves readability and allows tools 
    # to detect type-related bugs, but Python does not enforce it at runtime.
    """
    Define a seed DataFrame describing cluster centres.

    Parameters
    ----------
    column_specs : list of dict
        Each dict must contain:
        - 'name': str    – the column name
        - 'reps': list   – list of centre values, one per cluster

        Example:
        [
            {"name": "x", "reps": [0.0, 5.0, -5.0]},
            {"name": "y", "reps": [0.0, 5.0, -5.0]},
        ]

    Returns
    -------
    seed_df : pandas.DataFrame
        DataFrame with one row per cluster and one column per feature.
    """
    if not column_specs:
        raise ValueError("column_specs must be a non-empty list of dictionaries.")
    # Error-Handling: checks that column_specs is not empty

    # Check consistency of 'reps' lengths
    reps_lengths = [len(spec.get("reps", [])) for spec in column_specs] # list comprehension for each column/feature
    # we use .get instead of spec["reps"] to avoid KeyError if 'reps' key is missing
    if len(set(reps_lengths)) != 1: # makes the list of lengths into a set to remove duplicates, then checks if all lengths are the same
        raise ValueError("All 'reps' lists must have the same length (number of clusters).")
    # Error-Handling: validates that all 'reps' lists are the same length


    n_clusters = reps_lengths[0] # all of reps_lengths are the same anyway, so just take the first one
    data = {} # an empty dictionary
    for spec in column_specs: # iterate over each column/feature specification
        name = spec.get("name")
        reps = spec.get("reps")
        # saves the column name and reps list into variables for the first column
        if name is None or reps is None:
            raise ValueError("Each column_specs entry must have 'name' and 'reps' keys.")
        # Error-Handling: checks that 'name' and 'reps' actually exist
        if not isinstance(reps, Sequence):
            raise TypeError("'reps' must be a sequence of values.")
        # Error-Handling: checks that 'reps' is a sequence 
        # (list, tuple, numpy array, pandas series, range - something you can iterate over)
        if len(reps) != n_clusters:
            raise ValueError("All 'reps' lists must have the same length.")
        # Error-Handling: double-checks that current 'reps' list we are iterating over has the same as the first
        # we already checked this before - extra safeguard
        data[name] = list(reps)
        # makes sure that the reps are stored as lists then stores each feature's reps lists into the dictionary, data
        
# END OF FOR LOOP

# turns the dictionary, data, into a pandas DataFrame, seed_df
    seed_df = pd.DataFrame(data)
  # FIX: Used DataFrame(data) instead of orient="index".
    # The tests expect one row per cluster and one column per feature.
    # pd.DataFrame.from_dict(..., orient="index") transposed the structure,
    # producing shape (2, 3) instead of the required (3, 2).
    # Using pd.DataFrame(data) correctly treats each key as a column
    # and each 'reps' list as the column's values (rows).
    seed_df.index.name = "cluster_id"
    # names the index column as "cluster_id" for clarity
    return seed_df

# generates synthetic clustered data based on the cluster centres in seed_df
def simulate_data(
    seed_df: pd.DataFrame,
    n_points: int = 100,
    cluster_std: float = 1.0,
    # cluster_std must be numeric, not string. The default was previously "1.0" (a string), 
    # which caused a TypeError
    # Changing the default to float ensures correct numeric comparison.
    random_state: int | None = None,
) -> pd.DataFrame:
    # this function takes in the pandas DataFrame, seed_df, which describes the cluster centres, 
    # the number of points, n_points, with a default of 100,
    # the standard deviation of the clusters, cluster_std, witha default of 1.0
    # and a random seed, random_state, for reproducibility, optional
    
    # the function returns a pandas DataFrame containing the simulate data points
    """
    Simulate clustered data around the given cluster centres.

    Parameters
    ----------
    seed_df : pandas.DataFrame
        Rows represent cluster centres, columns represent features.
    n_points : int, default 100
        Total number of data points to simulate.
    cluster_std : float, default 1.0
        Standard deviation of Gaussian noise added around centres.
    random_state : int or None, default None
        Random seed for reproducibility.

    Returns
    -------
    data : pandas.DataFrame
        Simulated data with all original feature columns plus a 'true_cluster'
        column indicating the generating cluster.
    """
    if n_points <= 0:
        raise ValueError("n_points must be a positive integer.")
    # Error-Handling: checks that n_points is positive, thats the number of data points to simulate
    
    cluster_std = float(cluster_std)
    # Ensure cluster_std is always treated as a float.
    # This makes the function robust even if a user accidentally passes "1" or "0.5" as a string.
    if cluster_std <= 0:
        raise ValueError("cluster_std must be positive.")
    # Error-Handling: checks that cluster_std is positive, standard deviation cannot be negative or zero

    rng = np.random.RandomState(random_state)
    # set the random seed for reproducibility
    centres = seed_df.to_numpy(dtype=float)
    # converts the seed_df DataFrame into a numpy array of floats
    # each row is a cluster centre, each column is a feature
    n_clusters, n_features = centres.shape
    # defining n_clusters and n_features based on the shape of the centres array

    # Distribute points as evenly as possible across clusters.
    base = n_points // n_clusters # floor division to get the base number of points per cluster
    remainder = n_points % n_clusters # calculates any remainder points
    counts = np.full(n_clusters, base, dtype=int) 
    # creates an array of size n_clusters, filled with the base number of points per cluster
    counts[:remainder] += 1
    # adds 1 extra point to as many clusters as there are remainders

    records = [] # empty list called records
    for cluster_id, (centre, count) in enumerate(zip(centres, counts)): 
        # pairs each centre with its assigned number of points and iterates through this
        noise = rng.normal(loc=0.0, scale=cluster_std, size=(count, n_features)) # generating gaussian noise
        points = centre + noise # using broadcasting add the noise to the centre to create the points
        for point in points: # for each simulated point
            record = {col: val for col, val in zip(seed_df.columns, point)}
            # zip combines the column name in seed_df and the corresponding point number 
            # and the dictionary comprehension combines the features into a dictionary for each point
            record["true_cluster"] = cluster_id # change the cluster_id label to true_cluster
            records.append(record)
            # save the dictionaries to the list records

    data = pd.DataFrame.from_records(records)
    # converts the list of dictionaries—one per data point—into a full pandas DataFrame where each 
    # dictionary becomes a row and each key becomes a column
    return data