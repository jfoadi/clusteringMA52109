###
## cluster_maker
## James Foadi - University of Bath
## November 2025
###

from __future__ import annotations

from typing import List, Dict, Optional

import numpy as np
from sklearn.metrics import silhouette_score

from .algorithms import kmeans, sklearn_kmeans


def compute_inertia(
    X: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
) -> float:
    """
    Compute the within-cluster sum of squared distances (inertia).

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    labels : ndarray of shape (n_samples,)
    centroids : ndarray of shape (k, n_features)

    Returns
    -------
    inertia : float
    """
    if X.shape[0] != labels.shape[0]:
        raise ValueError("X and labels must have the same number of samples.")

    distances = X - centroids[labels]
    sq_dist = np.sum(distances ** 2)
    return float(sq_dist)


def silhouette_score_sklearn(
    X: np.ndarray,
    labels: np.ndarray,
) -> float:
    """
    Compute the silhouette score using scikit-learn.

    Returns
    -------
    score : float
    """
    # Silhouette is only defined when there are at least 2 clusters
    if len(np.unique(labels)) < 2:
        raise ValueError("Silhouette score requires at least 2 clusters.")
    return float(silhouette_score(X, labels))


def elbow_curve(
    X: np.ndarray,
    k_values: List[int],
    random_state: Optional[int] = None,
    use_sklearn: bool = True,
) -> Dict[int, float]:
    """
    Compute inertia values for multiple K values (elbow method).

    Parameters
    ----------
    X : ndarray
    k_values : list of int
    random_state : int or None
    use_sklearn : bool, default True
        If True, use scikit-learn KMeans; otherwise use manual kmeans.

    Returns
    -------
    inertia_dict : dict
        Mapping from k to inertia.
    """
    inertia_dict: Dict[int, float] = {}

    for k in k_values:
        if k <= 0:
            raise ValueError("All k values must be positive integers.")
        if use_sklearn:
            labels, centroids = sklearn_kmeans(X, k, random_state=random_state)
        else:
            labels, centroids = kmeans(X, k, random_state=random_state)
        inertia = compute_inertia(X, labels, centroids)
        inertia_dict[k] = inertia

    return inertia_dict