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

# Function for Task 6 PCA preprocessing
def pca_preprocess(
    data: pd.DataFrame,
    n_components: int | float = 0.95,
    standardize: bool = True
) -> pd.DataFrame:
    """
    Apply PCA to the numeric columns of a DataFrame.

    Parameters
    ----------
    data : pandas.DataFrame
        Input data (rows = samples, columns = features)
    n_components : int or float, default 0.95
        Number of components to keep, or fraction of variance explained.
    standardize : bool, default True
        Whether to standardize features before PCA.

    Returns
    -------
    transformed_data : pandas.DataFrame
        Data projected onto principal components.
    """
    numeric_data = data.select_dtypes(include=np.number)
    if numeric_data.empty:
        raise ValueError("No numeric columns available for PCA.")

    X = numeric_data.values.astype(float)

    # Standardize if needed
    if standardize:
        X = (X - X.mean(axis=0)) / X.std(axis=0, ddof=0)

    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)

    col_names = [f"PC{i+1}" for i in range(X_pca.shape[1])]
    return pd.DataFrame(X_pca, columns=col_names)
