# cluster_maker — short description

cluster_maker is a small Python package to help build, run and evaluate simple clustering workflows. The package is intended for teaching and quick prototyping rather than large-scale production use.

## Key features
- Build seed dataframes describing cluster centres and simulate noisy observations.
- Run clustering pipelines (preprocessing → algorithm → evaluation → plotting).
- Basic plotting helpers for quick 2D cluster visualisation and elbow plots.
- Small evaluation helpers to inspect clustering quality.
- Supportive utilities to select and validate features.

## Main modules and responsibilities

- cluster_maker.dataframe_builder
  - Purpose: define cluster centre structures and simulate datasets.
  - Key functions: define_dataframe_structure (builds a seed DataFrame where each row is a cluster centre),
    simulate_data (generates observations from those centres, returning a DataFrame with a `true_cluster` column).

- cluster_maker.algorithms
  - Purpose: implementations/wrappers for clustering algorithms (k-means, others).
  - Use: chosen by the orchestrator to fit cluster labels to data.

- cluster_maker.preprocessing
  - Purpose: common pre-processing steps (e.g., standardisation).
  - Use: applied before clustering to ensure features are on comparable scales.

- cluster_maker.interface
  - Purpose: top-level orchestration (high-level interface to run the whole pipeline).
  - Key function: run_clustering — loads input, selects features, runs pre-processing and algorithm, returns metrics and figures.

- cluster_maker.plotting_clustered
  - Purpose: plotting helpers for cluster visualisations and elbow plots.
  - Use: create 2D cluster scatter plots and elbow diagnostics for k selection.

- cluster_maker.evaluation
  - Purpose: compute clustering metrics (silhouette, within-cluster sum-of-squares, etc.) to assess results.

- cluster_maker.data_analyser (task extension)
  - Intended: summary/analysis helpers to compute column-wise numeric summaries (mean, sd, min/max, NA counts).

- cluster_maker.data_exporter (task extension)
  - Intended: write analysis results to CSV and to a simple human-readable text summary.

## Demo & tests
- demo/cluster_analysis.py demonstrates a simple 2D clustering pipeline using the package.
- Tests live in the `tests/` directory, run with pytest to validate components.
