###
## cluster_maker - cluster quality diagnostics
## James Foadi - University of Bath
## November 2025
###

from __future__ import annotations

from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


def cluster_quality_report(
    X: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
) -> Dict[str, float]:
    """
    Comprehensive statistical diagnostic for assessing cluster quality.

    Computes multiple cluster quality metrics:
    - Silhouette Score: Measures how similar points are to their own cluster
      compared to other clusters (range: -1 to 1, higher is better)
    - Davies-Bouldin Index: Ratio of within-cluster to between-cluster distances
      (lower is better, 0 is ideal)
    - Calinski-Harabasz Score: Ratio of between-cluster to within-cluster dispersion
      (higher is better)
    - Intra-cluster homogeneity: Average within-cluster distance
    - Inter-cluster separation: Average between-cluster distance

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Data points
    labels : ndarray of shape (n_samples,)
        Cluster labels for each point
    centroids : ndarray of shape (k, n_features)
        Cluster centroids

    Returns
    -------
    quality_metrics : dict
        Dictionary containing:
        - 'silhouette_score': Silhouette coefficient
        - 'davies_bouldin_index': Davies-Bouldin index
        - 'calinski_harabasz_score': Calinski-Harabasz score
        - 'intra_cluster_distance': Average within-cluster distance
        - 'inter_cluster_distance': Average between-cluster distance
        - 'compactness': Ratio of intra to inter-cluster distances

    Raises
    ------
    ValueError
        If there are fewer than 2 clusters or if X is empty.
    """
    if X.shape[0] == 0:
        raise ValueError("X must contain at least one sample.")
    
    n_clusters = len(np.unique(labels))
    if n_clusters < 2:
        raise ValueError("At least 2 clusters are required for quality assessment.")

    # Silhouette Score
    silhouette = silhouette_score(X, labels)

    # Davies-Bouldin Index
    davies_bouldin = davies_bouldin_score(X, labels)

    # Calinski-Harabasz Score
    calinski_harabasz = calinski_harabasz_score(X, labels)

    # Intra-cluster homogeneity (average within-cluster distance)
    intra_distances = []
    for cluster_id in np.unique(labels):
        cluster_mask = labels == cluster_id
        if cluster_mask.sum() > 0:
            cluster_points = X[cluster_mask]
            centroid = centroids[cluster_id]
            distances = np.linalg.norm(cluster_points - centroid, axis=1)
            intra_distances.extend(distances)
    
    intra_cluster_dist = np.mean(intra_distances) if intra_distances else 0.0

    # Inter-cluster separation (average distance between centroids)
    if n_clusters > 1:
        centroid_distances = cdist(centroids, centroids)
        # Get upper triangle (excluding diagonal)
        upper_triangle = centroid_distances[np.triu_indices_from(centroid_distances, k=1)]
        inter_cluster_dist = np.mean(upper_triangle) if len(upper_triangle) > 0 else 0.0
    else:
        inter_cluster_dist = 0.0

    # Compactness ratio (lower is better)
    if inter_cluster_dist > 0:
        compactness = intra_cluster_dist / inter_cluster_dist
    else:
        compactness = float('inf')

    quality_metrics = {
        'silhouette_score': float(silhouette),
        'davies_bouldin_index': float(davies_bouldin),
        'calinski_harabasz_score': float(calinski_harabasz),
        'intra_cluster_distance': float(intra_cluster_dist),
        'inter_cluster_distance': float(inter_cluster_dist),
        'compactness': float(compactness),
    }

    return quality_metrics


def cluster_quality_summary(quality_metrics: Dict[str, float]) -> str:
    """
    Generate a human-readable summary of cluster quality metrics.

    Parameters
    ----------
    quality_metrics : dict
        Dictionary of quality metrics (from cluster_quality_report)

    Returns
    -------
    summary : str
        Formatted string describing cluster quality
    """
    summary = "Cluster Quality Assessment\n"
    summary += "=" * 60 + "\n\n"

    summary += f"Silhouette Score:           {quality_metrics['silhouette_score']:.4f}\n"
    summary += "  (Range: -1 to 1, higher is better)\n\n"

    summary += f"Davies-Bouldin Index:       {quality_metrics['davies_bouldin_index']:.4f}\n"
    summary += "  (Lower is better, 0 is ideal)\n\n"

    summary += f"Calinski-Harabasz Score:    {quality_metrics['calinski_harabasz_score']:.4f}\n"
    summary += "  (Higher is better)\n\n"

    summary += f"Intra-cluster Distance:     {quality_metrics['intra_cluster_distance']:.4f}\n"
    summary += "  (Average within-cluster distance)\n\n"

    summary += f"Inter-cluster Distance:     {quality_metrics['inter_cluster_distance']:.4f}\n"
    summary += "  (Average between-cluster distance)\n\n"

    summary += f"Compactness Ratio:          {quality_metrics['compactness']:.4f}\n"
    summary += "  (Intra/Inter ratio, lower is better)\n\n"

    return summary
