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
from .data_analyser import calculate_descriptive_statistics, calculate_correlation, numeric_summary
from .data_exporter import export_to_csv, export_formatted, export_summary_csv, export_summary_text

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
)

# --- Plotting ---
from .plotting_clustered import plot_clusters_2d, plot_elbow

# --- Cluster quality diagnostics ---
from .cluster_quality import cluster_quality_report, cluster_quality_summary

# --- High-level interface ---
from .interface import run_clustering


__all__ = [
    # Data generation
    "define_dataframe_structure",
    "simulate_data",

    # Analysis
    "calculate_descriptive_statistics",
    "calculate_correlation",
    "numeric_summary",

    # Export
    "export_to_csv",
    "export_formatted",
    "export_summary_csv",
    "export_summary_text",

    # Preprocessing
    "select_features",
    "standardise_features",

    # Algorithms
    "kmeans",
    "sklearn_kmeans",
    "init_centroids",
    "assign_clusters",
    "update_centroids",

    # Evaluation
    "compute_inertia",
    "silhouette_score_sklearn",
    "elbow_curve",

    # Plotting
    "plot_clusters_2d",
    "plot_elbow",

    # Cluster quality diagnostics
    "cluster_quality_report",
    "cluster_quality_summary",

    # High-level orchestration
    "run_clustering",
]