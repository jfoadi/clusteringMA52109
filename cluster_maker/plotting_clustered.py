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
    if n_clusters <= 10:
        base_cmap = plt.get_cmap("tab10")
        colors = [base_cmap(i) for i in range(n_clusters)]
        cmap = plt.matplotlib.colors.ListedColormap(colors)
    else:
        # Fallback for many clusters
        cmap = "viridis"

    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create scatter plot with discrete colormap
    scatter = ax.scatter(
        X[:, 0], 
        X[:, 1], 
        c=labels, 
        cmap=cmap, 
        alpha=0.7,
        s=50,
        edgecolors='white',
        linewidth=0.5,
        vmin=-0.5, 
        vmax=n_clusters - 0.5
    )

    if centroids is not None:
        ax.scatter(
            centroids[:, 0],
            centroids[:, 1],
            marker="X",
            s=300,
            linewidths=3,
            color="black",
            edgecolors="white",
            label="Centroids",
            zorder=10
        )
        ax.legend(loc='best', frameon=True, shadow=True)

    ax.set_xlabel("Feature 1", fontsize=11, fontweight='bold')
    ax.set_ylabel("Feature 2", fontsize=11, fontweight='bold')
    
    if title:
        ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
    
    # Add subtle grid
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

    # Discrete ticks for the colorbar
    cbar = fig.colorbar(scatter, ax=ax, label="Cluster", ticks=np.arange(n_clusters))
    cbar.ax.set_ylabel("Cluster", fontsize=10, fontweight='bold')
    
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

    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot with enhanced styling
    ax.plot(
        k_values, 
        inertias, 
        marker="o", 
        linewidth=2.5,
        markersize=8,
        color='#2E86AB',
        markerfacecolor='#A23B72',
        markeredgecolor='white',
        markeredgewidth=1.5
    )
    
    ax.set_xlabel("Number of clusters (k)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Inertia (WCSS)", fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
    
    # Enhanced grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    
    # Set integer ticks for k-axis
    ax.set_xticks(k_values)
    
    fig.tight_layout()
    return fig, ax