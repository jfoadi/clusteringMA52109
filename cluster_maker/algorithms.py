###
## cluster_maker
## James Foadi - University of Bath
## November 2025
###

from __future__ import annotations

from typing import Tuple, Optional

import numpy as np
from sklearn.cluster import KMeans

# Randomly selects k data points from `X` (without replacement) to use as the initial centroids 
# for k-means clustering
def init_centroids(
    X: np.ndarray,
    k: int,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    Initialise centroids by randomly sampling points from X without replacement.
    """
    if k <= 0:
        raise ValueError("k must be a positive integer.")
    # Gets the number of rows in X (i.e., how many data points you have)
    n_samples = X.shape[0]
    if k > n_samples:
        raise ValueError("k cannot be larger than the number of samples.")

    rng = np.random.RandomState(random_state)
    # choose k numbers from 1 to n_samples (number of data points), without replacement
    indices = rng.choice(n_samples, size=k, replace=False)
    # return only those rows from X, thats your centroids
    return X[indices]


# Computes the distance from each data point to each centroid 
# and assigns each point to the closest centroid
def assign_clusters(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """
    Assign each sample to the nearest centroid (Euclidean distance).
    """
    # X: (n_samples, n_features)
    # centroids: (k, n_features)
    # Broadcast to compute distances
    # For every sample and every centroid, compute (sample − centroid)
    diff = X[:, np.newaxis, :] - centroids[np.newaxis, :, :]
    # Takes the L2 norm across the feature dimension
    # Converts the (n_samples, k, n_features) tensor into (n_samples, k)
    distances = np.linalg.norm(diff, axis=2)  # (n_samples, k)
    # Pick the closest centroid for each sample
    labels = np.argmin(distances, axis=1)
    return labels


# Recomputes each centroid by taking the mean of all points assigned to that cluster, 
# and if a cluster has no points, randomly reinitialises its centroid
def update_centroids(
    X: np.ndarray,
    labels: np.ndarray,
    k: int,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    Update centroids by taking the mean of points in each cluster.
    If a cluster becomes empty, re-initialise its centroid randomly from X.
    """
    n_features = X.shape[1] # gets number of features
    new_centroids = np.zeros((k, n_features), dtype=float) # prepare a zeros array for new centroids
    rng = np.random.RandomState(random_state) # Create random generator for empty-cluster handling
# Loop over each cluster
    for cluster_id in range(k):
        # Identify samples belonging to this cluster using a boolean array selecting points assigned to cluster_id
        mask = labels == cluster_id 
        if not np.any(mask):
            # Empty cluster: re-initialise randomly
            idx = rng.randint(0, X.shape[0])
            new_centroids[cluster_id] = X[idx]
        else:
            # Takes the average of all samples assigned to the cluster, sets that as the updated centroid position
            new_centroids[cluster_id] = X[mask].mean(axis=0)
    return new_centroids

# Runs a complete manual K-means clustering algorithm: 
# initialises centroids → assigns points → updates centroids → 
# repeats until convergence → returns final labels and centroids
def kmeans(
    X: np.ndarray,
    k: int,
    max_iter: int = 300,
    tol: float = 1e-4,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple manual K-means implementation.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    k : int
        Number of clusters.
    max_iter : int, default 300
        Maximum number of iterations.
    tol : float, default 1e-4
        Convergence tolerance on centroid movement.
    random_state : int or None

    Returns
    -------
    labels : ndarray of shape (n_samples,)
    centroids : ndarray of shape (k, n_features)
    """
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a NumPy array.")
    # ERROR-HANDLING: if X is not a NumPy array
    
    # initialize centroids using init_centroids created earlier
    centroids = init_centroids(X, k, random_state=random_state)
    for _ in range(max_iter): # for loop, iterates
        # assign clusters using assign_clusters created earlier
        labels = assign_clusters(X, centroids)
        # update_centroids using funtion created earlier
        new_centroids = update_centroids(X, labels, k, random_state=random_state)
        # Checking for convergence: calculate distance from new centroids to old centroids
        shift = np.linalg.norm(new_centroids - centroids)
        # replace old centroids with new centroids
        centroids = new_centroids
        # if the distance from old to new centroids is small, break for loop
        if shift < tol:
            break
    # Ensures labels match the final centroids
    labels = assign_clusters(X, centroids)
    return labels, centroids



# Runs scikit-learn’s built-in KMeans algorithm and returns the resulting cluster labels and centroids
def sklearn_kmeans(
    X: np.ndarray,
    k: int,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Thin wrapper around scikit-learn's KMeans.

    Returns
    -------
    labels : ndarray of shape (n_samples,)
    centroids : ndarray of shape (k, n_features)
    """
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a NumPy array.")
    
    # Create the scikit-learn KMeans model
    model = KMeans(
        n_clusters=k,
        random_state=random_state,
        n_init=10,
        # n_init=10: runs KMeans 10 times with different centroid seeds 
        # internally (scikit-learn picks the best result)
    )
    model.fit(X) # Fit model to data; Performs the entire K-means procedure
    
    # Extract labels and centroids
    labels = model.labels_
    centroids = model.cluster_centers_
    return labels, centroids