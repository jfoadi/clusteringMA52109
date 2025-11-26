###
## cluster_maker: auto-demo with data generation + clustering
## Modified for mock exam – University of Bath
## This version generates its own CSV, then runs clustering on it.
###

from __future__ import annotations

import os
import sys
import pandas as pd

from cluster_maker import (
    define_dataframe_structure,
    simulate_data,
    run_clustering,
    select_features,
)

OUTPUT_DIR = "demo_output"
DATA_DIR = "data"
GENERATED_CSV = os.path.join(DATA_DIR, "generated_demo_data.csv")


def generate_csv() -> str:
    """
    Generate a synthetic CSV dataset using cluster_maker's own tools.
    Returns the path to the generated CSV file.
    """

    print("=== Generating synthetic CSV using cluster_maker ===")

    # Step 1: Define cluster centres
    column_specs = [
        {"name": "x", "reps": [0.0, 5.0, -5.0]},
        {"name": "y", "reps": [0.0, 5.0, -5.0]},
        {"name": "z", "reps": [1.0, 2.0, 3.0]},
    ]

    seed_df = define_dataframe_structure(column_specs)
    print("Seed DataFrame:")
    print(seed_df)

    # Step 2: Simulate synthetic data
    data = simulate_data(seed_df, n_points=300, cluster_std=1.0, random_state=42)
    print("\nSample of generated data:")
    print(data.head())

    # Ensure data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)

    # Step 3: Save to CSV
    data.to_csv(GENERATED_CSV, index=False)
    print(f"\nCSV saved to {GENERATED_CSV}")

    print("=== Finished generating synthetic CSV ===\n")
    return GENERATED_CSV


def main() -> None:
    print("=== cluster_maker DEMO: full automatic analysis ===\n")

    print("Step 1: Creating synthetic data...")
    input_path = generate_csv()

    print("Step 2: Loading generated data...")
    df = pd.read_csv(input_path)
    print("Loaded successfully.")
    print(f"Columns detected: {list(df.columns)}")

    # Identify numeric columns
    numeric_cols = [
        col for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col])
    ]

    if len(numeric_cols) < 2:
        print("ERROR: The generated dataset unexpectedly has fewer than 2 numeric columns.")
        sys.exit(1)

    # Select first two numeric columns for 2D clustering
    feature_cols = numeric_cols[:2]
    print(f"Using feature columns: {feature_cols}")

    # Validate using package function
    select_features(df, feature_cols)

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\nStep 3: Running clustering using run_clustering()")
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
    for k, v in result["metrics"].items():
        print(f"  {k}: {v}")

    print("\nStep 4: Saving plots and output files...")
    cluster_plot_path = os.path.join(OUTPUT_DIR, "cluster_plot.png")
    elbow_plot_path = os.path.join(OUTPUT_DIR, "elbow_plot.png")

    result["fig_cluster"].savefig(cluster_plot_path, dpi=150)
    if result["fig_elbow"] is not None:
        result["fig_elbow"].savefig(elbow_plot_path, dpi=150)

    print("Files saved:")
    print(f"  - clustered_data.csv")
    print(f"  - {cluster_plot_path}")
    if result["fig_elbow"] is not None:
        print(f"  - {elbow_plot_path}")

    print("\n=== End of automatic demo ===")


if __name__ == "__main__":
    main()
