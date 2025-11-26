###
## cluster_maker: demo for cluster analysis
## James Foadi - University of Bath
## November 2025
###

"""
Demo script for cluster_maker package: clustering analysis workflow.

This script demonstrates the complete clustering workflow including:
- Data loading and validation
- Feature selection
- K-means clustering
- Metric computation (inertia, silhouette score)
- Visualization (cluster plot, elbow curve)
- Result export to CSV

Usage:
    python demo/cluster_analysis.py <input_csv_file>

Example:
    python demo/cluster_analysis.py demo/sample_data.csv

Advanced Usage with PCA (Task 6):
    To use PCA for dimensionality reduction before clustering, modify the
    run_clustering call to include:
        use_pca=True,
        n_components=2,
    
    This will apply Principal Component Analysis to reduce high-dimensional
    data to 2 components before clustering.
"""

from __future__ import annotations

import os
import sys
import pandas as pd

from cluster_maker import run_clustering, select_features

OUTPUT_DIR = "demo_output"


def main(args: list[str]) -> None:
    print("=== cluster_maker demo: clustering analysis ===\n")

    # Require exactly one argument: the CSV file path
    if len(args) != 2:
        print("ERROR: Incorrect number of arguments provided.")
        print("Usage: python demo/demo_cluster_analysis.py [input_csv_file]")
        sys.exit(1)

    # Input CSV file
    input_path = args[1]
    print(f"Input CSV file: {input_path}")

    # Check file exists
    if not os.path.exists(input_path):
        print(f"\nERROR: The file '{input_path}' does not exist.")
        sys.exit(1)

    # Load data so we can inspect numeric columns
    print("\nLoading data to inspect available features...")
    df = pd.read_csv(input_path)
    print("Data loaded successfully.")
    print(f"Number of rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    # Identify numeric columns
    numeric_cols = [
        col for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col])
    ]

    if len(numeric_cols) < 2:
        print("\nERROR: Not enough numeric columns for 2D clustering.")
        print(f"Numeric columns found: {numeric_cols}")
        sys.exit(1)

    # Take the first two numeric columns
    feature_cols = numeric_cols[:2]
    print(f"Chosen numeric feature columns for clustering: {feature_cols}")
    print("-" * 60)

    # Validate feature columns using the package function
    try:
        _ = select_features(df, feature_cols)
    except Exception as exc:
        print(f"\nERROR validating features with select_features:\n{exc}")
        sys.exit(1)

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Run the orchestrator
    print("Running clustering with run_clustering(...)")
    result = run_clustering(
        input_path=input_path,
        feature_cols=feature_cols,
        algorithm="kmeans",
        k=3,
        standardise=True,
        output_path=os.path.join(OUTPUT_DIR, "clustered_data.csv"),
        random_state=42,
        compute_elbow=True,
    )

    print("\nClustering completed.")
    print("Metrics:")
    for key, value in result["metrics"].items():
        print(f"  {key}: {value}")
    print("-" * 60)

    # Save plots
    cluster_plot_path = os.path.join(OUTPUT_DIR, "cluster_plot.png")
    elbow_plot_path = os.path.join(OUTPUT_DIR, "elbow_plot.png")

    print(f"Saving 2D cluster plot to:\n  {cluster_plot_path}")
    result["fig_cluster"].savefig(cluster_plot_path, dpi=150)

    if result["fig_elbow"] is not None:
        print(f"Saving elbow plot to:\n  {elbow_plot_path}")
        result["fig_elbow"].savefig(elbow_plot_path, dpi=150)
    else:
        print("No elbow plot was generated (fig_elbow is None).")

    print("\nClustered data saved to:")
    print(f"  - {os.path.join(OUTPUT_DIR, 'clustered_data.csv')}")
    print("Plots saved to:")
    print(f"  - {cluster_plot_path}")
    if result["fig_elbow"] is not None:
        print(f"  - {elbow_plot_path}")

    # ============================================================
    # ADVANCED FEATURE DEMONSTRATION: PCA (Task 6)
    # ============================================================
    print("\n" + "=" * 60)
    print("ADVANCED FEATURE: Demonstrating PCA preprocessing")
    print("=" * 60)
    print("\nRunning clustering WITH PCA (dimensionality reduction)...")
    
    result_pca = run_clustering(
        input_path=input_path,
        feature_cols=feature_cols,
        algorithm="kmeans",
        k=3,
        standardise=True,
        use_pca=True,           # <-- PCA ENABLED (Task 6)
        n_components=2,         # <-- Reduce to 2 principal components
        output_path=os.path.join(OUTPUT_DIR, "clustered_data_pca.csv"),
        random_state=42,
        compute_elbow=False,    # Skip elbow for brevity
    )

    print("\nPCA-based clustering completed.")
    print("Metrics:")
    for key, value in result_pca["metrics"].items():
        print(f"  {key}: {value}")
    
    # Save PCA plot
    pca_plot_path = os.path.join(OUTPUT_DIR, "cluster_plot_pca.png")
    print(f"\nSaving PCA cluster plot to:\n  {pca_plot_path}")
    result_pca["fig_cluster"].savefig(pca_plot_path, dpi=150)
    
    print("\nPCA demonstration complete!")
    print(f"PCA-transformed data saved to: {os.path.join(OUTPUT_DIR, 'clustered_data_pca.csv')}")
    print(f"PCA plot saved to: {pca_plot_path}")
    print("\nNOTE: When PCA is used, features are transformed to Principal Components (PC1, PC2).")
    print("This is useful for high-dimensional data or visualization purposes.")

    print("\n=== End of demo ===")


if __name__ == "__main__":
    main(sys.argv)