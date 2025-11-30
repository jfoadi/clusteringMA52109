###
## cluster_maker
## Yas Akilakulasingam - University of Bath
## November 2025
###


"""
Cluster stability diagnostic for evaluating robustness of clustering.

This module introduces a statistically meaningful extension that measures
how consistent the clustering output is under repeated perturbations.
"""

from __future__ import annotations
import numpy as np
from .algorithms import kmeans

# Measures how **stable** clustering results are by repeatedly adding small noise to the data, 
# re-running KMeans, fixing label switching, and computing how often each pair of points 
# ends up in the same cluster.
def cluster_stability_score(
    X: np.ndarray,
    k: int,
    n_runs: int = 20,
    noise_scale: float = 0.05,
    random_state: int | None = None,
    algorithm: str = "kmeans",  
) -> float:
    """
    Compute clustering stability while correcting for label switching.

    Stability is measured by repeatedly adding small noise to the dataset,
    reclustering it, aligning labels between runs, and computing how often 
    pairs of points appear in the same cluster.

    Produces a score in [0,1], where 1 = perfectly stable.
    """

    # ------------------------------------------------------------
    # ERROR HANDLING
    # ------------------------------------------------------------

    # X must be NumPy array
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a NumPy array.")

    n_samples = X.shape[0]
    if n_samples < 2:
        raise ValueError("X must contain at least 2 samples.")

    # k must be valid
    if not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer.")
    if k > n_samples:
        raise ValueError("k cannot be larger than the number of samples.")

    # n_runs must be >= 1
    if n_runs < 1:
        raise ValueError("n_runs must be at least 1.")

    # noise_scale must be >= 0
    if noise_scale < 0:
        raise ValueError("noise_scale must be non-negative.")

    # algorithm must be valid
    if algorithm not in ("kmeans", "sklearn_kmeans"):
        raise ValueError("algorithm must be 'kmeans' or 'sklearn_kmeans'.")

    # ------------------------------------------------------------
    # Choose clustering function
    # ------------------------------------------------------------
    if algorithm == "kmeans":
        cluster_func = kmeans
    else:
        cluster_func = sklearn_kmeans  # imported from algorithms.py

    rng = np.random.RandomState(random_state)

    # ------------------------------------------------------------
    # Initialise co-occurrence matrix
    # ------------------------------------------------------------
    co_matrix = np.zeros((n_samples, n_samples))

    # ------------------------------------------------------------
    # Base clustering (reference run)
    # ------------------------------------------------------------
    try:
        base_labels, base_centroids = cluster_func(X, k=k, random_state=random_state)
    except Exception as exc:
        raise RuntimeError(f"Clustering failed in base run: {exc}")

    # Align labels by sorting centroids
    order = np.lexsort((base_centroids[:, 1], base_centroids[:, 0]))
    label_map = {old: new for new, old in enumerate(order)}
    aligned_base = np.array([label_map[l] for l in base_labels])

    # Update co-occurrence matrix
    for i in range(n_samples):
        for j in range(n_samples):
            if aligned_base[i] == aligned_base[j]:
                co_matrix[i, j] += 1

    # ------------------------------------------------------------
    # Perturbation runs (n_runs - 1 times)
    # ------------------------------------------------------------
    for _ in range(n_runs - 1):

        # Add Gaussian noise
        noise = rng.normal(scale=noise_scale, size=X.shape)
        X_noisy = X + noise

        # Cluster noisy data
        try:
            labels, centroids = cluster_func(X_noisy, k=k, random_state=random_state)
        except Exception as exc:
            raise RuntimeError(f"Clustering failed during noisy run: {exc}")

        # Align labels again using sorted centroids
        order = np.lexsort((centroids[:, 1], centroids[:, 0]))
        label_map = {old: new for new, old in enumerate(order)}
        aligned = np.array([label_map[l] for l in labels])

        # Update co-occurrence matrix
        for i in range(n_samples):
            for j in range(n_samples):
                if aligned[i] == aligned[j]:
                    co_matrix[i, j] += 1

    # ------------------------------------------------------------
    # Final Stability Score
    # ------------------------------------------------------------

    # Average co-occurrence across runs
    stability_matrix = co_matrix / n_runs

    # Compute off-diagonal mean
    stability = (
        np.sum(stability_matrix) - np.trace(stability_matrix)
    ) / (n_samples * (n_samples - 1))

    return float(stability)
