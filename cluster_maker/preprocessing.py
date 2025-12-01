###
## cluster_maker
## James Foadi - University of Bath
## November 2025
###

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def select_features(data: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """
    Select a subset of columns to use as features, ensuring they are numeric.

    Parameters
    ----------
    data : pandas.DataFrame
    feature_cols : list of str
        Column names to select.

    Returns
    -------
    X_df : pandas.DataFrame
        DataFrame containing only the selected feature columns.

    Raises
    ------
    KeyError
        If any requested column is missing.
    TypeError
        If any selected column is non-numeric.
    """
    missing = [col for col in feature_cols if col not in data.columns]
    if missing:
        raise KeyError(f"The following feature columns are missing: {missing}")

    X_df = data[feature_cols].copy()

    non_numeric = [
        col for col in X_df.columns
        if not pd.api.types.is_numeric_dtype(X_df[col])
    ]
    if non_numeric:
        raise TypeError(f"The following feature columns are not numeric: {non_numeric}")

    return X_df


def standardise_features(X: np.ndarray) -> np.ndarray:
    """
    Standardise features to zero mean and unit variance.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)

    Returns
    -------
    X_scaled : ndarray of shape (n_samples, n_features)
    """
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a NumPy array.")
    scaler = StandardScaler()
    return scaler.fit_transform(X)


def apply_pca(X: np.ndarray, n_components: int = 2):
    """
    Apply PCA dimensionality reduction using scikit-learn.

    Parameters
    ----------
    X : ndarray (n_samples, n_features)
        Numeric data matrix.
    n_components : int, default 2
        Number of principal components to retain.

    Returns
    -------
    X_pca : ndarray (n_samples, n_components)
        Transformed data in the reduced PCA space.
    explained_variance_ratio : ndarray
        Variance explained by each selected component.

    Raises
    ------
    TypeError
        If X is not a numpy array.
    ValueError
        If X is not 2D numeric, or if n_components is out of range.
    """

    # --- Error handling ---
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a NumPy array.")
    if X.ndim != 2:
        raise ValueError("X must be a 2D numeric array.")
    if not np.issubdtype(X.dtype, np.number):
        raise ValueError("X must contain numeric values only.")

    n_samples, n_features = X.shape

    if not isinstance(n_components, int) or n_components <= 0:
        raise ValueError("n_components must be a positive integer.")
    if n_components > n_features:
        raise ValueError("n_components cannot exceed number of features.")

    # Apply PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    return X_pca, pca.explained_variance_ratio_