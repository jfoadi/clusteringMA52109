###
## cluster_maker
## James Foadi - University of Bath
## November 2025
###

from __future__ import annotations

from typing import List, Dict, Optional

import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score

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


def compute_davies_bouldin(
    X: np.ndarray,
    labels: np.ndarray,
    return_details: bool = False,
) -> dict | float:
    """
    Compute the Davies–Bouldin Index (DBI) with validation and optional detail.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    labels : ndarray of shape (n_samples,)
        Cluster labels.
    return_details : bool, default False
        If True, return a dictionary with additional diagnostic information.

    Returns
    -------
    float or dict
        If return_details=False:
            dbi : float
        If return_details=True:
            {
                "dbi": float,
                "clusters": int,
                "valid": bool
            }

    Notes
    -----
    DBI is undefined for a single cluster. In that case, the function
    returns np.nan and marks valid=False in details mode.
    """

    # --------- Validate inputs ----------
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a NumPy array.")

    if not isinstance(labels, np.ndarray):
        raise TypeError("labels must be a NumPy array.")

    if X.ndim != 2:
        raise ValueError("X must be a 2D array of shape (n_samples, n_features).")

    if labels.ndim != 1:
        raise ValueError("labels must be a 1D array of shape (n_samples,).")

    if X.shape[0] != labels.shape[0]:
        raise ValueError("X and labels must contain the same number of samples.")

    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    # Need at least two clusters
    if n_clusters < 2:
        if return_details:
            return {"dbi": np.nan, "clusters": n_clusters, "valid": False}
        return np.nan

    # --------- Compute DBI ----------
    dbi = davies_bouldin_score(X, labels)

    if return_details:
        return {"dbi": float(dbi), "clusters": n_clusters, "valid": True}

    return float(dbi)