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

def apply_pca(
    df: pd.DataFrame,
    n_components: int = 2,
    standardise: bool = True,
) -> pd.DataFrame:
    """
    Apply Principal Component Analysis (PCA) to the numeric columns of a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing numeric data.
    n_components : int, default 2
        Number of principal components to keep.
    standardise : bool, default True
        Whether to standardise features before PCA.

    Returns
    -------
    pca_df : pandas.DataFrame
        DataFrame containing the principal component scores.
        Columns are named PC1, PC2, ..., PCn.

    Notes
    -----
    - Non-numeric columns are ignored.
    - Raises ValueError if there are no numeric columns.
    """
    # Keep only numeric columns
    numeric_df = df.select_dtypes(include="number")

    if numeric_df.empty:
        raise ValueError("No numeric columns available for PCA.")

    data = numeric_df.to_numpy(dtype=float)

    # Optional standardisation: zero mean, unit variance
    if standardise:
        means = data.mean(axis=0)
        stds = data.std(axis=0, ddof=0)
        stds[stds == 0] = 1.0  # avoid division by zero
        data = (data - means) / stds

    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(data)

    component_names = [f"PC{i+1}" for i in range(n_components)]
    pca_df = pd.DataFrame(transformed, columns=component_names)

    return pca_df
