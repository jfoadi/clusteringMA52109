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

def apply_pca(X: np.ndarray, n_components: int) -> np.ndarray:
    """
    Apply Principal Component Analysis (PCA) and project data onto the
    specified number of components.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    n_components : int
        The number of principal components to keep.

    Returns
    -------
    X_pca : ndarray of shape (n_samples, n_components)
        The data projected onto the principal components.
    """
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a NumPy array.")
    if n_components <= 0:
        raise ValueError("n_components must be a positive integer.")
    if n_components > X.shape[1]:
        raise ValueError("n_components cannot exceed the number of features in X.")

    # PCA should typically be done after standardisation (which is handled 
    # outside this function in run_clustering).
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    
    return X_pca