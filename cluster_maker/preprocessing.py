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


def apply_pca(data: pd.DataFrame, n_components: int = 2) -> pd.DataFrame:
    """
    Apply PCA to the data.

    Parameters
    ----------
    data : pandas.DataFrame
        Input data (numeric).
    n_components : int, default 2
        Number of principal components to keep.

    Returns
    -------
    pca_df : pandas.DataFrame
        Transformed data with columns "PC1", "PC2", etc.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")

    # Ensure data is numeric
    numeric_data = data.select_dtypes(include=[np.number])
    if numeric_data.shape[1] < n_components:
        raise ValueError(
            f"n_components ({n_components}) cannot be greater than the number of numeric features ({numeric_data.shape[1]})."
        )

    pca = PCA(n_components=n_components)
    transformed_data = pca.fit_transform(numeric_data)

    columns = [f"PC{i+1}" for i in range(n_components)]
    return pd.DataFrame(transformed_data, columns=columns, index=data.index)


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
    ValueError
        If any requested column is missing.
    TypeError
        If any selected column is non-numeric.
    """
    missing = [col for col in feature_cols if col not in data.columns]
    if missing:
        raise ValueError(f"The following feature columns are missing: {missing}")

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