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

    # Determine number of clusters for discrete colormap
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    # Create a discrete colormap using the first n_clusters colors of tab10
    # Note: tab10 has 10 colors. If n_clusters > 10, we might need a larger palette,
    # but for this assignment k is usually small.
    if n_clusters <= 10:
        base_cmap = plt.get_cmap("tab10")
        colors = [base_cmap(i) for i in range(n_clusters)]
        cmap = plt.matplotlib.colors.ListedColormap(colors)
    else:
        # Fallback for many clusters
        cmap = "viridis"

    fig, ax = plt.subplots()
    
    # We need to ensure the scatter uses the discrete map correctly
    # We can use a BoundaryNorm or just rely on the ListedColormap with integer labels
    # if we normalize to [0, n_clusters-1]
    
    scatter = ax.scatter(
        X[:, 0], 
        X[:, 1], 
        c=labels, 
        cmap=cmap, 
        alpha=0.8,
        vmin=-0.5, 
        vmax=n_clusters - 0.5
    )

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

    # Discrete ticks for the colorbar
    cbar = fig.colorbar(scatter, ax=ax, label="Cluster label", ticks=np.arange(n_clusters))
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