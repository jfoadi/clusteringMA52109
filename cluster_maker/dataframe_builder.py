###
## cluster_maker
## James Foadi - University of Bath
## November 2025
###

# cluster_maker/dataframe_builder.py

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Dict, Any

def define_dataframe_structure(column_specs: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Define a seed DataFrame describing cluster centres.

    Parameters
    ----------
    column_specs : List[Dict[str, Any]]
        A list of dictionaries defining column names and their centre values.

    Returns
    -------
    pd.DataFrame
        A DataFrame where each row represents a cluster centre.
    """
    # Error Handling (10% marks): Check for empty input
    if not column_specs:
        raise ValueError("column_specs must be a non-empty list.")

    # Error Handling: Check if all 'reps' have the same length
    reps_lengths = [len(spec.get("reps", [])) for spec in column_specs]
    if len(set(reps_lengths)) != 1:
        raise ValueError("All 'reps' lists must have the same length.")

    # Logic: Convert list-of-dicts to a single dict for Pandas
    data_dict = {}
    for spec in column_specs:
        col_name = spec["name"]
        col_values = spec["reps"]
        data_dict[col_name] = col_values
    
    return pd.DataFrame(data_dict)

def simulate_data(
    seed_df: pd.DataFrame,
    n_points: int = 100,
    cluster_std: float = 1.0,
    random_state: int | None = None,
) -> pd.DataFrame:
    """
    Simulate clustered data based on the seed DataFrame.

    Parameters
    ----------
    seed_df : pd.DataFrame
        The dataframe containing cluster centres.
    n_points : int
        Total data points to generate.
    cluster_std : float
        The standard deviation (spread) of the clusters.
    random_state : int, optional
        Seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Simulated data including a 'true_cluster' column.
    """
    # Error Handling (10% marks)
    if n_points <= 0:
        raise ValueError("n_points must be positive.")
    
    # Reproducibility (Marking requirement)
    rng = np.random.RandomState(random_state)
    
    centres = seed_df.to_numpy(dtype=float)
    n_clusters, n_features = centres.shape
    feature_names = seed_df.columns.tolist()

    # Distribute points evenly
    base = n_points // n_clusters
    remainder = n_points % n_clusters
    counts = np.full(n_clusters, base, dtype=int)
    counts[:remainder] += 1

    records = []
    
    # Loop through each cluster centre
    for cluster_id, (centre, count) in enumerate(zip(centres, counts)):
        # Generate noise
        noise = rng.normal(loc=0.0, scale=cluster_std, size=(count, n_features))
        points = centre + noise
        
        # Save points
        for point in points:
            row = dict(zip(feature_names, point))
            # CRITICAL FIX: The test fails without this line!
            row["true_cluster"] = cluster_id 
            records.append(row)

    return pd.DataFrame(records)