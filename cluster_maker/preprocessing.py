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
        raise ValueError(f"Required feature column(s) not found in data: {missing}")

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
    Applies Principal Component Analysis (PCA) to reduce the dimensionality 
    of the feature matrix X.

    Parameters
    ----------
    X : numpy.ndarray
        The feature matrix (rows are samples, columns are features).
    n_components : int
        The number of principal components to retain (must be <= X.shape[1]).

    Returns
    -------
    X_reduced : numpy.ndarray
        The data transformed into the principal component subspace.
        
    Raises
    ------
    ValueError
        If n_components is invalid (e.g., > number of features).
    """
    n_features = X.shape[1]
    if n_components <= 0 or n_components > n_features:
        raise ValueError(
            f"n_components ({n_components}) must be > 0 and <= number of features ({n_features})."
        )

    # Initialize and fit PCA
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X)
    
    # Store explained variance ratio in the PCA object if needed for diagnostics,
    # but the function only returns the transformed data for pipeline integration.
    
    return X_reduced