# cluster_maker - Package Description

## Overview
`cluster_maker` is an educational Python package for simulating clustered datasets and performing clustering analyses.  
It is designed for practicals and exams, providing a simple interface to learn clustering concepts and workflows.

## Main Features
- **Seed DataFrame:** define cluster centres for simulation using `define_dataframe_structure`.
- **Data Simulation:** generate clustered datasets around defined centres using `simulate_data`.
- **Preprocessing:** feature selection and standardisation (module: `preprocessing.py`).
- **Clustering:**
  - Manual K-means implementation (`algorithms.py`).
  - scikit-learn KMeans wrapper (`algorithms.py`).
- **Evaluation:**
  - Compute inertia (within-cluster sum of squares).
  - Silhouette score.
  - Elbow curve for choosing the number of clusters (`evaluation.py`).
- **Plotting:**
  - 2D scatter plots of clusters (`plotting_clustered.py`).
  - Optional centroids and elbow plots.
- **High-level interface:** `run_clustering` for an end-to-end workflow (`interface.py`).
- **Demo scripts:** example usage in the `demo/` folder.

## Key Modules
- `dataframe_builder.py` – define cluster centres (`define_dataframe_structure`) and simulate data (`simulate_data`).
- `data_analyser.py` – descriptive statistics and correlations.
- `data_exporter.py` – export summaries to CSV or formatted text.
- `preprocessing.py` – feature selection and standardisation functions.
- `algorithms.py` – manual K-means and scikit-learn KMeans wrapper.
- `evaluation.py` – compute cluster quality metrics and elbow curves.
- `plotting_clustered.py` – visualisation of clusters and evaluation plots.
- `interface.py` – high-level `run_clustering` function for a complete workflow.

## Example Usage
```python
from cluster_maker.dataframe_builder import define_dataframe_structure, simulate_data
from cluster_maker.interface import run_clustering

# Define cluster centres
seed_df = define_dataframe_structure([
    {"name": "x", "reps": [0, 5, -5]},
    {"name": "y", "reps": [0, 5, -5]}
])

# Simulate data
data = simulate_data(seed_df, n_points=100, cluster_std=1.0)

# Run clustering and evaluation
run_clustering(data, n_clusters=3, random_state=42)
