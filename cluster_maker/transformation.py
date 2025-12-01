## This module is intended to transform data through PCA.

import pandas as pd
from sklearn.decomposition import PCA


def apply_pca(data: pd.DataFrame, n_components: int = 2) -> pd.DataFrame:
    """
    Apply PCA to reduce the dimensionality of the dataset.

    Parameters
    ----------
    data : pandas.DataFrame
        The input data with numeric features.
    n_components : int
        The number of principal components to keep.

    Returns
    -------
    transformed_data : pandas.DataFrame
        DataFrame containing the principal components.
    """
    if n_components <= 0 or n_components > data.shape[1]:
        raise ValueError(
            f"n_components ({n_components}) must be between "
            f"1 and the number of features in the data ({data.shape[1]}).")

    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(data)
    pc_df = pd.DataFrame(
        data=principal_components,
        columns=[f'PC{i+1}' for i in range(n_components)]
    )
    return pc_df