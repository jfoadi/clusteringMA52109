###
## cluster_maker
## James Foadi - University of Bath
## November 2025
###

from __future__ import annotations

from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt


def plot_clusters_2d(X, labels, centroids=None, title="Cluster plot"):
    plt.figure(figsize=(10, 7))
    plt.style.use("seaborn-v0_8")

    scatter = plt.scatter(
        X[:, 0], 
        X[:, 1], 
        c=labels, 
        cmap="viridis",
        s=70,
        alpha=0.8,
        edgecolors="black",
        linewidth=0.4
    )

    if centroids is not None:
        plt.scatter(
            centroids[:, 0], 
            centroids[:, 1],
            c="white",
            edgecolors="black",
            s=300,
            marker="X",
            linewidth=1.5,
            label="Centroids"
        )

    plt.title(title, fontsize=18, fontweight="bold")
    plt.xlabel("Feature 1", fontsize=14)
    plt.ylabel("Feature 2", fontsize=14)
    plt.grid(alpha=0.3)
    plt.colorbar(scatter, label="Cluster label")

    if centroids is not None:
        plt.legend()

    fig = plt.gcf()
    plt.close()
    return fig



def plot_elbow(k_values, inertias, title="Elbow Curve"):
    plt.figure(figsize=(10, 7))
    plt.style.use("seaborn-v0_8")

    plt.plot(k_values, inertias, marker="o", markersize=10, linewidth=2)
    plt.title(title, fontsize=18, fontweight="bold")
    plt.xlabel("Number of clusters (k)", fontsize=14)
    plt.ylabel("Inertia", fontsize=14)
    plt.grid(alpha=0.3)

    fig = plt.gcf()
    plt.close()
    return fig

