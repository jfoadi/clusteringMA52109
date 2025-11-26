###
## cluster_maker
## James Foadi - University of Bath
## November 2025
###

from __future__ import annotations

from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

from .preprocessing import select_features, standardise_features, apply_pca
from .algorithms import kmeans, sklearn_kmeans
from .evaluation import compute_inertia, elbow_curve, silhouette_score_sklearn
from .plotting_clustered import plot_clusters_2d, plot_elbow
from .data_exporter import export_to_csv


def run_clustering(
    input_path: str,
    feature_cols: List[str],
    algorithm: str = "kmeans",
    k: int = 3,
    standardise: bool = True,
    use_pca: bool = False,
    n_components: int = 2,
    output_path: Optional[str] = None,
    random_state: Optional[int] = None,
    compute_elbow: bool = False,
    elbow_k_values: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    High-level function to run the full clustering workflow.

    Steps:
    1. Load data from CSV
    2. Select feature columns
    3. Optionally standardise features
    4. Optionally apply PCA
    5. Run the chosen clustering algorithm
    6. Compute evaluation metrics
    7. Generate plots
    8. Optionally write labelled data to CSV

    Parameters
    ----------
    input_path : str
        Path to the input CSV file.
    feature_cols : list of str
        Names of feature columns to use.
    algorithm : {"kmeans", "sklearn_kmeans"}, default "kmeans"
    k : int, default 3
        Number of clusters.
    standardise : bool, default True
    use_pca : bool, default False
        If True, apply PCA before clustering.
    n_components : int, default 2
        Number of PCA components to keep if use_pca is True.
    output_path : str or None, default None
        If provided, the input data with cluster labels will be saved to this CSV.
    random_state : int or None, default None
    compute_elbow : bool, default False
        If True, compute inertia for multiple k values.
    elbow_k_values : list of int or None, default None
        k-values for elbow curve. If None and compute_elbow is True, defaults
        to range 1..(k+5).

    Returns
    -------
    result : dict
        Dictionary containing:
        - "data": DataFrame with added "cluster" column
        - "labels": ndarray of cluster labels
        - "centroids": ndarray of cluster centroids
        - "metrics": dict with "inertia" and optional "silhouette"
        - "fig_cluster": Figure for the cluster plot
        - "fig_elbow": Figure for the elbow plot or None
        - "elbow_inertias": dict mapping k -> inertia (if computed)
    """
    # Load data
    df = pd.read_csv(input_path)

    # Select and optionally standardise features
    X_df = select_features(df, feature_cols)
    X = X_df.to_numpy(dtype=float)

    if standardise:
        X = standardise_features(X)

    # Apply PCA if requested
    if use_pca:
        # We need to convert back to DataFrame for apply_pca (as per my implementation)
        # or update apply_pca to accept numpy array.
        # My apply_pca takes DataFrame.
        # But X is numpy array here.
        # Let's convert X back to DataFrame or update apply_pca.
        # Converting X back to DataFrame is easier here since we have column names.
        X_df_scaled = pd.DataFrame(X, columns=feature_cols)
        X_pca_df = apply_pca(X_df_scaled, n_components=n_components)
        X = X_pca_df.to_numpy()
        # Update feature names for plotting
        feature_cols = list(X_pca_df.columns)

    # Run clustering
    if algorithm == "kmeans":
        labels, centroids = kmeans(X, k=k, random_state=random_state)
    elif algorithm == "sklearn_kmeans":
        labels, centroids = sklearn_kmeans(X, k=k, random_state=random_state)
    else:
        raise ValueError(f"Unknown algorithm '{algorithm}'. Use 'kmeans' or 'sklearn_kmeans'.")

    # Compute metrics
    inertia = compute_inertia(X, labels, centroids)
    metrics: Dict[str, Any] = {"inertia": inertia}

    try:
        sil = silhouette_score_sklearn(X, labels)
    except ValueError:
        sil = None
    metrics["silhouette"] = sil

    # Add labels to DataFrame
    df = df.copy()
    df["cluster"] = labels

    # Export if requested
    if output_path is not None:
        export_to_csv(df, output_path, delimiter=",", include_index=False)

    # Plot clusters (2D)
    # Note: If PCA was used, X has PC columns. feature_cols was updated.
    fig_cluster, _ = plot_clusters_2d(X, labels, centroids=centroids, title="Cluster plot")

    # Optional elbow curve
    fig_elbow = None
    elbow_inertias: Optional[Dict[int, float]] = None
    if compute_elbow:
        if elbow_k_values is None:
            max_k = max(2, k + 5)
            elbow_k_values = list(range(1, max_k + 1))
        elbow_inertias = elbow_curve(
            X,
            k_values=elbow_k_values,
            random_state=random_state,
            use_sklearn=(algorithm == "sklearn_kmeans"),
        )
        fig_elbow, _ = plot_elbow(
            elbow_k_values,
            [elbow_inertias[val] for val in elbow_k_values],
        )

    result: Dict[str, Any] = {
        "data": df,
        "labels": labels,
        "centroids": centroids,
        "metrics": metrics,
        "fig_cluster": fig_cluster,
        "fig_elbow": fig_elbow,
        "elbow_inertias": elbow_inertias,
    }
    return result