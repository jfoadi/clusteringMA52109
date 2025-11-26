###
## cluster_maker – PCA tools
## MA52109 – Mock Exam Extension
## November 2025
###

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def apply_pca(X: np.ndarray, n_components: int = 2) -> np.ndarray:
    """
    Apply PCA dimensionality reduction.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input feature matrix.
    n_components : int, default 2
        Number of PCA components to keep.

    Returns
    -------
    X_reduced : ndarray of shape (n_samples, n_components)
        PCA-transformed data.
    """
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a NumPy array.")

    if n_components <= 0:
        raise ValueError("n_components must be a positive integer.")

    pca = PCA(n_components=n_components)
    return pca.fit_transform(X)


def plot_pca_2d(X_reduced, labels=None, title="PCA – 2D Projection"):
    plt.figure(figsize=(10, 7))
    plt.style.use("seaborn-v0_8")

    if labels is None:
        plt.scatter(X_reduced[:, 0], X_reduced[:, 1], s=60, alpha=0.7)
    else:
        plt.scatter(
            X_reduced[:, 0],
            X_reduced[:, 1],
            c=labels,
            cmap="viridis",
            s=70,
            edgecolors="black",
            linewidth=0.5
        )

    plt.title(title, fontsize=18, fontweight="bold")
    plt.xlabel("Principal Component 1", fontsize=14)
    plt.ylabel("Principal Component 2", fontsize=14)
    plt.grid(alpha=0.3)

    fig = plt.gcf()
    plt.close()
    return fig
