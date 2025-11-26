###
## cluster_maker
## James Foadi - University of Bath
## November 2025
###

from __future__ import annotations

from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

from .preprocessing import select_features, standardise_features, pca_transform   # <-- add pca_transform
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
    output_path: Optional[str] = None,
    random_state: Optional[int] = None,
    compute_elbow: bool = False,
    elbow_k_values: Optional[List[int]] = None,
    use_pca: bool = False,   # <-- NEW explicit parameter
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
    """
    # Load data
    df = pd.read_csv(input_path)

    # Select and optionally standardise features
    X_df = select_features(df, feature_cols)
    X = X_df.to_numpy(dtype=float)

    if standardise:
        X = standardise_features(X)

    # Optional PCA reduction
    if use_pca:
        X = pca_transform(X, n_components=2, random_state=random_state)

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
