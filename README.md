# cluster_maker

`cluster_maker` is a small educational Python package for simulating clustered
datasets and running clustering analyses with a simple, user-friendly interface.

It is designed for practicals and exams where students are given an incomplete
or faulty version of the package and asked to debug or extend it.

## Quick Start

After installation, try the demo scripts:

```bash
# Main clustering demo (Task 2)
python demo/cluster_analysis.py demo/sample_data.csv

# Numeric analysis demo (Task 4)
python demo/analyse_from_csv.py demo/sample_data.csv
```

Check `demo_output/` for results including CSV files and visualizations.

## Allowed libraries

The package only uses:

- Python standard library  
- NumPy  
- pandas  
- matplotlib  
- SciPy  
- scikit-learn  

No other third-party libraries are required.

## Main features

### Data Generation
- Define a **seed DataFrame** describing cluster centres  
- Simulate clustered data around these centres with customizable noise

### Data Analysis (NEW - Task 3)
- Compute **descriptive statistics** (mean, std, min, max)
- Calculate **correlations** between features
- Generate **numeric summaries** with missing value counts

### Data Export (NEW - Task 3)
- Export to **CSV** format
- Export to **formatted text** files
- Combined export of summary statistics

### Preprocessing
- **Feature selection** with validation
- **Standardisation** (zero mean, unit variance)
- **PCA (Principal Component Analysis)** for dimensionality reduction (NEW - Task 6)

### Clustering Algorithms
- Simple **manual K-means** implementation (educational)
- Scikit-learn **KMeans** wrapper (production-ready)

### Evaluation Metrics
- **Inertia** (within-cluster sum of squares)  
- **Silhouette score** (cluster quality measure)
- **Elbow curve** for optimal K selection

### Visualization
- 2D cluster scatter plots with centroids
- Discrete colormaps for clear cluster distinction
- Elbow curve plots for K-selection
- Professional styling with proper labels and legends

### High-Level Interface
- **`run_clustering`** function: one-line clustering workflow
- Supports PCA, standardization, multiple algorithms
- Automatic metric computation and visualization generation

### Demo Scripts
- `cluster_analysis.py` - Complete clustering workflow demonstration
- `analyse_from_csv.py` - CSV analysis and summary export

## Package root directory structure

- `cluster_maker/`
  - `dataframe_builder.py` – build seed DataFrame and simulate clustered data  
  - `data_analyser.py` – descriptive statistics, correlation, and numeric summaries
  - `data_exporter.py` – CSV and formatted text export with combined export
  - `preprocessing.py` – feature selection, standardisation, and PCA
  - `algorithms.py` – manual K-means and scikit-learn KMeans wrapper  
  - `evaluation.py` – inertia, silhouette, elbow curve  
  - `plotting_clustered.py` – 2D cluster plots and elbow plots  
  - `interface.py` – high-level `run_clustering` function with PCA support
- `demo/` – example scripts demonstrating package functionality
- `tests/` – comprehensive unit tests using `unittest` framework
  - `test_dataframe_builder.py` - Tests for data generation, analysis, and PCA
  - `test_interface_and_export.py` - Tests for interface error handling and export

## Installation (local use)

From the root directory of the project, run:

```bash
pip install -e .
```

This installs the package in editable mode, meaning you can modify the files
and re-run tests or demos without reinstalling.

## Running Tests

Run all tests:
```bash
pytest
```

Run specific test files:
```bash
pytest tests/test_dataframe_builder.py
pytest tests/test_interface_and_export.py
```

## Notes on pyproject.toml and the *.egg-info directory

This project includes a small file named `pyproject.toml`.
You do not need to open or edit it. Its only purpose is to tell Python/pip
that this folder is a valid installable package. Without it, the command
`pip install -e .` would fail.

When you run the installation command, pip automatically creates a directory
called something like:

`cluster_maker.egg-info/`

This folder contains package metadata used internally by Python (file lists,
version information, etc.). It is generated automatically and should not be
edited. You can safely ignore it during the mock-practical.
