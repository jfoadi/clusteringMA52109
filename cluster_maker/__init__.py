"""
cluster_maker

An educational Python package for generating synthetic clustered data,
running clustering algorithms, evaluating results, and producing
user-friendly plots. Designed for practicals and exams where students
work with an incomplete or faulty version of the package and must fix it.

Allowed libraries:
- Python standard library
- numpy
- pandas
- matplotlib
- scipy
- scikit-learn
"""

# --- Data generation & basic analysis ---
from .dataframe_builder import define_dataframe_structure, simulate_data
# Updated for Task 3
from .data_analyser import calculate_descriptive_statistics, calculate_correlation, summarise_numeric_data
# Updated for Task 3
from .data_exporter import export_to_csv, export_formatted, export_summary_report

# --- Preprocessing ---
from .preprocessing import select_features, standardise_features

# --- Clustering algorithms ---
from .algorithms import (
    kmeans,
    sklearn_kmeans,
    init_centroids,
    assign_clusters,
    update_centroids,
)

# --- Evaluation ---
from .evaluation import (
    compute_inertia,
    silhouette_score_sklearn,
    elbow_curve,
    calculate_pca_explained_variance, # Task 6
)

# --- Plotting ---
from .plotting_clustered import plot_clusters_2d, plot_elbow

# --- High-level interface ---
from .interface import run_clustering


__all__ = [
    # Data generation
    "define_dataframe_structure",
    "simulate_data",

    # Analysis (Task 3)
    "calculate_descriptive_statistics",
    "calculate_correlation",
    "summarise_numeric_data",

    # Export (Task 3)
    "export_to_csv",
    "export_formatted",
    "export_summary_report",

    # Preprocessing
    "select_features",
    "standardise_features",

    # Algorithms
    "kmeans",
    "sklearn_kmeans",
    "init_centroids",
    "assign_clusters",
    "update_centroids",

    # Evaluation (Task 6)
    "compute_inertia",
    "silhouette_score_sklearn",
    "elbow_curve",
    "calculate_pca_explained_variance",

    # Plotting
    "plot_clusters_2d",
    "plot_elbow",

    # High-level orchestration
    "run_clustering",
]