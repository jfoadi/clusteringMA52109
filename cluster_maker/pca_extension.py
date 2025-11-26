"""
cluster_maker.pca_extension

PCA-based preprocessing extension for cluster_maker.

Provides a simple wrapper around sklearn.decomposition.PCA that:
- fits PCA to a NumPy array X,
- returns the transformed array and the fitted PCA object,
- includes basic input validation and helpful docstrings.

Only uses allowed libraries.
"""

from __future__ import annotations

from typing import Tuple, Optional

import numpy as np
from sklearn.decomposition import PCA

def pca_extension(
    X: np.ndarray,
    n_components: int,
    whiten: bool = False,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, PCA]:
    """
    Fit PCA to X and return the transformed data along with the PCA object.

    Parameters
    ----------
    X : np.ndarray
        The input data array of shape (n_samples, n_features).
    n_components : int
        The number of principal components to compute.
    whiten : bool, optional
        If True, the components will be whitened. Default is False.
    random_state : Optional[int], optional
        Seed for the random number generator. Default is None.
    
    Returns
    -------
    X_pca : np.ndarray, shape (n_samples, n_components)
        The transformed data array of shape (n_samples, n_components).
    pca : sklearn.decomposition.PCA
        The fitted PCA object.
    """

    if not isinstance(X, np.ndarray):
        raise TypeError("Input X must be a NumPy array.")
    if X.ndim != 2:
        raise ValueError("Input X must be a 2D array.")
    n_samples, n_features = X.shape
    if not isinstance(n_components, int) or n_components <= 0:
        raise ValueError("n_components must be a positive integer.")
    if n_components > n_features:
        raise ValueError("n_components cannot be greater than the number of features.")

    # Use sklearn's PCA to fit and transform the data
    pca = PCA(n_components=n_components, whiten=whiten, random_state=random_state)
    X_pca = pca.fit_transform(X)

    return X_pca, pca
    
    