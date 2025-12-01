###
## cluster_maker  PCA Analysis Module Extension
## Mock Practical MA52109 Task 6)
## November 2025
###

from __future__ import annotations

from typing import Tuple, Optional, Dict, Any
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def compute_pca(
    X: np.ndarray,
    n_components: Optional[int] = None,
    standardize: bool = True,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, PCA, Optional[StandardScaler]]:
    """
    Perform Principal Component Analysis on the input data.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Input data matrix
    n_components : int or None, default None
        Number of components to keep. If None, all components are kept.
    standardize : bool, default True
        Whether to standardize the features before PCA (recommended).
    random_state : int or None, default None
        Random seed for reproducibility.

    Returns
    -------
    X_pca : np.ndarray of shape (n_samples, n_components)
        Transformed data in principal component space.
    pca_model : sklearn.decomposition.PCA
        Fitted PCA model for future transformations.
    scaler : sklearn.preprocessing.StandardScaler or None
        Fitted scaler if standardize=True, else None.
    """
    print("Starting PCA computation...")
    
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a NumPy array.")
    
    if X.ndim != 2:
        raise ValueError("X must be a 2-dimensional array.")
    
    n_samples, n_features = X.shape
    print(f"Input data shape: {n_samples} samples, {n_features} features")
    
    if n_samples < 2:
        raise ValueError("PCA requires at least 2 samples.")
    
    if n_features < 2:
        raise ValueError("PCA requires at least 2 features.")
    
    # Standardize the data if requested
    scaler = None
    if standardize:
        print("Standardizing features (zero mean, unit variance)...")
        scaler = StandardScaler()
        X_processed = scaler.fit_transform(X)
        print("Feature standardization completed")
    else:
        X_processed = X.copy()
        print("Skipping feature standardization")
    
    # Perform PCA
    print(f"Performing PCA with {n_components if n_components else 'all'} components...")
    pca_model = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca_model.fit_transform(X_processed)
    
    print("PCA computation completed successfully!")
    print(f"Output shape: {X_pca.shape}")
    
    return X_pca, pca_model, scaler


def explained_variance_analysis(pca_model: PCA) -> Dict[str, Any]:
    """
    Analyze and return explained variance information from PCA model.

    Parameters
    ----------
    pca_model : sklearn.decomposition.PCA
        Fitted PCA model.

    Returns
    -------
    variance_info : dict
        Dictionary containing:
        - 'explained_variance_ratio': array of variance ratios per component
        - 'cumulative_variance_ratio': cumulative variance explained
        - 'n_components_95': number of components for 95% variance
        - 'n_components_90': number of components for 90% variance
    """
    print("Analyzing explained variance...")
    
    explained_variance_ratio = pca_model.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    # Find number of components for common variance thresholds
    n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
    n_components_90 = np.argmax(cumulative_variance >= 0.90) + 1
    
    variance_info = {
        'explained_variance_ratio': explained_variance_ratio,
        'cumulative_variance_ratio': cumulative_variance,
        'n_components_95': n_components_95,
        'n_components_90': n_components_90,
        'total_variance_explained': cumulative_variance[-1] if len(cumulative_variance) > 0 else 0
    }
    
    print(f"Variance analysis completed:")
    print(f"  - Total variance explained: {variance_info['total_variance_explained']:.3f}")
    print(f"  - Components for 90% variance: {n_components_90}")
    print(f"  - Components for 95% variance: {n_components_95}")
    
    return variance_info


def plot_pca_variance(
    pca_model: PCA,
    title: str = "PCA Explained Variance"
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create a scree plot showing explained variance by principal components.

    Parameters
    ----------
    pca_model : sklearn.decomposition.PCA
        Fitted PCA model.
    title : str, default "PCA Explained Variance"
        Plot title.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    print("Creating PCA variance plot...")
    
    variance_info = explained_variance_analysis(pca_model)
    explained_variance_ratio = variance_info['explained_variance_ratio']
    cumulative_variance = variance_info['cumulative_variance_ratio']
    
    n_components = len(explained_variance_ratio)
    component_numbers = np.arange(1, n_components + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Scree plot (individual variance)
    ax1.bar(component_numbers, explained_variance_ratio, alpha=0.7, color='skyblue')
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Explained Variance Ratio')
    ax1.set_title('Scree Plot - Individual Variance')
    ax1.grid(True, alpha=0.3)
    
    # Cumulative variance plot
    ax2.plot(component_numbers, cumulative_variance, 'o-', color='orange', linewidth=2)
    ax2.axhline(y=0.95, color='r', linestyle='--', alpha=0.7, label='95% variance')
    ax2.axhline(y=0.90, color='g', linestyle='--', alpha=0.7, label='90% variance')
    ax2.axhline(y=0.80, color='b', linestyle='--', alpha=0.7, label='80% variance')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Explained Variance')
    ax2.set_title('Cumulative Variance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    fig.tight_layout()
    
    print("PCA variance plot created successfully!")
    return fig, (ax1, ax2)


def pca_clustering_workflow(
    X: np.ndarray,
    n_pca_components: Optional[int] = None,
    clustering_algorithm: str = "kmeans",
    k: int = 3,
    standardize: bool = True,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Complete workflow: PCA preprocessing followed by clustering.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Input data matrix.
    n_pca_components : int or None, default None
        Number of PCA components. If None, determined automatically.
    clustering_algorithm : str, default "kmeans"
        Clustering algorithm to use ("kmeans" or "sklearn_kmeans").
    k : int, default 3
        Number of clusters.
    standardize : bool, default True
        Whether to standardize before PCA.
    random_state : int or None, default None
        Random seed for reproducibility.

    Returns
    -------
    result : dict
        Dictionary containing:
        - 'X_original': Original input data
        - 'X_pca': PCA-transformed data
        - 'pca_model': Fitted PCA model
        - 'variance_info': Explained variance analysis
        - 'clustering_result': Results from clustering on PCA data
        - 'fig_variance': Figure with variance plots
    """
    print("=" * 60)
    print("PCA-CLUSTERING WORKFLOW: STARTING")
    print("=" * 60)
    
    from cluster_maker.algorithms import kmeans, sklearn_kmeans
    from cluster_maker.evaluation import compute_inertia, silhouette_score_sklearn
    
    # Step 1: Perform PCA
    print("\nSTEP 1: Performing PCA preprocessing...")
    X_pca, pca_model, scaler = compute_pca(
        X, 
        n_components=n_pca_components,
        standardize=standardize,
        random_state=random_state
    )
    
    # Step 2: Analyze explained variance
    print("\nSTEP 2: Analyzing explained variance...")
    variance_info = explained_variance_analysis(pca_model)
    
    # Step 3: Perform clustering on PCA-transformed data
    print("\nSTEP 3: Performing clustering on PCA components...")
    if clustering_algorithm == "kmeans":
        labels, centroids = kmeans(X_pca, k=k, random_state=random_state)
    elif clustering_algorithm == "sklearn_kmeans":
        labels, centroids = sklearn_kmeans(X_pca, k=k, random_state=random_state)
    else:
        raise ValueError(f"Unknown algorithm: {clustering_algorithm}")
    
    # Step 4: Compute clustering metrics
    print("\nSTEP 4: Computing clustering metrics...")
    inertia = compute_inertia(X_pca, labels, centroids)
    
    try:
        silhouette = silhouette_score_sklearn(X_pca, labels)
    except ValueError:
        silhouette = None
    
    # Step 5: Create visualization
    print("\nSTEP 5: Creating variance visualization...")
    fig_variance, _ = plot_pca_variance(pca_model, "PCA Analysis for Clustering")
    
    # Compile results
    result = {
        'X_original': X,
        'X_pca': X_pca,
        'pca_model': pca_model,
        'scaler': scaler,
        'variance_info': variance_info,
        'clustering_result': {
            'labels': labels,
            'centroids': centroids,
            'metrics': {
                'inertia': inertia,
                'silhouette': silhouette
            }
        },
        'fig_variance': fig_variance
    }
    
    print("\n" + "=" * 60)
    print("PCA-CLUSTERING WORKFLOW: COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("WORKFLOW SUMMARY:")
    print(f"  - Original features: {X.shape[1]}")
    print(f"  - PCA components: {X_pca.shape[1]}")
    print(f"  - Variance explained: {variance_info['total_variance_explained']:.3f}")
    print(f"  - Clustering algorithm: {clustering_algorithm}")
    print(f"  - Number of clusters: {k}")
    print(f"  - Clustering inertia: {inertia:.4f}")
    if silhouette is not None:
        print(f"  - Clustering silhouette: {silhouette:.4f}")
    print("=" * 60)
    
    return result