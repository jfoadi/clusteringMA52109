###
## cluster_maker
## James Foadi - University of Bath
## November 2025
###

from __future__ import annotations

from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt


def plot_clusters_2d(
    X: np.ndarray,
    labels: np.ndarray,
    centroids: Optional[np.ndarray] = None,
    title: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot clustered data in 2D using the first two features.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    labels : ndarray of shape (n_samples,)
    centroids : ndarray of shape (k, n_features) or None
    title : str or None

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    if X.shape[1] < 2:
        raise ValueError("X must have at least 2 features for a 2D plot.")

    fig, ax = plt.subplots()
    ax.grid(True, linestyle='--', alpha=0.6)
    scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap="tab10", alpha=0.8)

    if centroids is not None:
        ax.scatter(
            centroids[:, 0],
            centroids[:, 1],
            marker="x",
            s=200,
            linewidths=2,
            color="black",
            label="Centroids",
        )
        ax.legend()

    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    if title:
        ax.set_title(title)

    fig.colorbar(scatter, ax=ax, label="Cluster label")
    fig.tight_layout()
    return fig, ax


def plot_elbow(
    k_values: List[int],
    inertias: List[float],
    title: str = "Elbow Curve",
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot inertia vs k (elbow method).

    Parameters
    ----------
    k_values : list of int
    inertias : list of float
    title : str, default "Elbow Curve"

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    if len(k_values) != len(inertias):
        raise ValueError("k_values and inertias must have the same length.")

    fig, ax = plt.subplots()
    ax.plot(k_values, inertias, marker="o")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Inertia")
    ax.set_title(title)
    ax.grid(True)
    fig.tight_layout()
    return fig, ax