###
## cluster_maker: demo for cluster analysis
## James Foadi - University of Bath
## November 2025
###

from __future__ import annotations

import os
import sys
import pandas as pd
import numpy as np

# --- PATH FIX ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from cluster_maker import run_clustering, select_features
from cluster_maker.dataframe_builder import define_dataframe_structure, simulate_data

OUTPUT_DIR = "demo_output"
TEMP_INPUT_PATH = os.path.join(OUTPUT_DIR, "temp_data_input.csv")

def print_header(title: str):
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80 + "\n")

def main() -> None:
    print_header("cluster_maker Demo: Algorithm Comparison")
    print("This script simulates data and compares two clustering approaches:")
    print(" 1. K-Means (Centroid-based)")
    print(" 2. Agglomerative Hierarchical (Connectivity-based)")
    print("-" * 80)

    # --- 1. Generate Data ---
    print("\n[Step 1] Generating Synthetic Data...")
    n_points = 300
    n_clusters = 3
    seed_specs = [
        {"name": "FeatureA", "reps": [0.0, 5.0, -5.0]},
        {"name": "FeatureB", "reps": [0.0, 5.0, -5.0]},
    ]
    seed_df = define_dataframe_structure(seed_specs)
    df = simulate_data(seed_df, n_points=n_points, cluster_std=0.8, random_state=42)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(TEMP_INPUT_PATH, index=False)
    print(f" > Generated {n_points} samples with {n_clusters} true clusters.")

    # --- 2. Run Both Algorithms ---
    feature_cols = ["FeatureA", "FeatureB"]
    algorithms = ["kmeans", "agglomerative"]
    results = {}

    print("\n[Step 2] Running Comparative Analysis...")
    
    for algo in algorithms:
        print(f" > Running {algo.capitalize()}...", end=" ")
        try:
            # We calculate Elbow only for KMeans as it relies on inertia
            compute_elbow = (algo == "kmeans")
            
            res = run_clustering(
                input_path=TEMP_INPUT_PATH,
                feature_cols=feature_cols,
                algorithm=algo,
                k=n_clusters,
                standardise=True,
                output_path=os.path.join(OUTPUT_DIR, f"clustered_{algo}.csv"),
                compute_elbow=compute_elbow
            )
            results[algo] = res
            print("Done.")
            
            # Save specific plot
            plot_path = os.path.join(OUTPUT_DIR, f"plot_{algo}.png")
            res["fig_cluster"].savefig(plot_path, dpi=150)
            
        except Exception as e:
            print(f"\nFAILED: {e}")

    # --- 3. Comparative Results Table ---
    print_header("Comparative Results")
    
    # Define table layout
    row_fmt = "{:<15} | {:<15} | {:<15} | {:<15}"
    print(row_fmt.format("Metric", "K-Means", "Agglomerative", "Winner"))
    print("-" * 70)

    # Extract metrics
    km_metrics = results["kmeans"]["metrics"]
    agg_metrics = results["agglomerative"]["metrics"]

    # Compare Inertia
    km_in = km_metrics.get("inertia", float('inf'))
    agg_in = agg_metrics.get("inertia", float('inf'))
    winner_in = "K-Means" if km_in < agg_in else "Agglomerative"
    print(row_fmt.format("Inertia", f"{km_in:.2f}", f"{agg_in:.2f}", winner_in))

    # Compare Silhouette
    km_sil = km_metrics.get("silhouette", 0)
    agg_sil = agg_metrics.get("silhouette", 0)
    winner_sil = "K-Means" if km_sil > agg_sil else "Agglomerative"
    print(row_fmt.format("Silhouette", f"{km_sil:.4f}", f"{agg_sil:.4f}", winner_sil))

    print("-" * 70)
    print("(Note: Lower Inertia is better; Higher Silhouette is better)")

    # --- 4. File Summary ---
    print("\n[Step 4] Saved Outputs")
    print(f"All files saved to: {OUTPUT_DIR}")
    print(f" - clustered_kmeans.csv / clustered_agglomerative.csv")
    print(f" - plot_kmeans.png      / plot_agglomerative.png")
    
    if results["kmeans"]["fig_elbow"]:
         results["kmeans"]["fig_elbow"].savefig(os.path.join(OUTPUT_DIR, "elbow_kmeans.png"))
         print(" - elbow_kmeans.png")

    print("\nComparison completed successfully.")

if __name__ == "__main__":
    main()