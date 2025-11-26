###
## cluster_maker - tests for cluster quality diagnostics
## James Foadi - University of Bath
## November 2025
###

import unittest

import numpy as np
import pandas as pd

from cluster_maker.cluster_quality import cluster_quality_report, cluster_quality_summary


class TestClusterQualityDiagnostics(unittest.TestCase):
    """Test cluster quality diagnostic functions."""

    def setUp(self):
        """Set up test data."""
        # Create well-separated clusters
        np.random.seed(42)
        
        # Cluster 1: centered at (0, 0)
        cluster1 = np.random.randn(50, 2) * 0.3 + np.array([0, 0])
        
        # Cluster 2: centered at (5, 5)
        cluster2 = np.random.randn(50, 2) * 0.3 + np.array([5, 5])
        
        # Cluster 3: centered at (0, 5)
        cluster3 = np.random.randn(50, 2) * 0.3 + np.array([0, 5])
        
        self.X = np.vstack([cluster1, cluster2, cluster3])
        self.labels = np.array([0] * 50 + [1] * 50 + [2] * 50)
        
        # Centroids
        self.centroids = np.array([
            np.mean(cluster1, axis=0),
            np.mean(cluster2, axis=0),
            np.mean(cluster3, axis=0),
        ])

    def test_cluster_quality_report_returns_dict(self):
        """Test that cluster_quality_report returns a dictionary."""
        result = cluster_quality_report(self.X, self.labels, self.centroids)
        self.assertIsInstance(result, dict)

    def test_cluster_quality_report_has_required_keys(self):
        """Test that cluster_quality_report includes all required metrics."""
        result = cluster_quality_report(self.X, self.labels, self.centroids)
        
        required_keys = {
            'silhouette_score',
            'davies_bouldin_index',
            'calinski_harabasz_score',
            'intra_cluster_distance',
            'inter_cluster_distance',
            'compactness',
        }
        self.assertEqual(set(result.keys()), required_keys)

    def test_cluster_quality_report_metric_ranges(self):
        """Test that metrics are within expected ranges."""
        result = cluster_quality_report(self.X, self.labels, self.centroids)
        
        # Silhouette score should be between -1 and 1
        self.assertGreaterEqual(result['silhouette_score'], -1.0)
        self.assertLessEqual(result['silhouette_score'], 1.0)
        
        # Davies-Bouldin index should be non-negative
        self.assertGreaterEqual(result['davies_bouldin_index'], 0.0)
        
        # Calinski-Harabasz score should be positive
        self.assertGreater(result['calinski_harabasz_score'], 0.0)
        
        # Distances should be non-negative
        self.assertGreaterEqual(result['intra_cluster_distance'], 0.0)
        self.assertGreaterEqual(result['inter_cluster_distance'], 0.0)
        self.assertGreaterEqual(result['compactness'], 0.0)

    def test_cluster_quality_report_with_two_clusters(self):
        """Test cluster_quality_report with minimum number of clusters."""
        X = self.X[:100]  # Use only 2 clusters
        labels = self.labels[:100]
        centroids = self.centroids[:2]
        
        result = cluster_quality_report(X, labels, centroids)
        self.assertIsInstance(result, dict)
        self.assertGreater(result['silhouette_score'], -1.0)

    def test_cluster_quality_report_raises_on_single_cluster(self):
        """Test that cluster_quality_report raises error with single cluster."""
        labels = np.zeros(self.X.shape[0], dtype=int)
        
        with self.assertRaises(ValueError):
            cluster_quality_report(self.X, labels, self.centroids)

    def test_cluster_quality_report_raises_on_empty_data(self):
        """Test that cluster_quality_report raises error with empty data."""
        X_empty = np.empty((0, 2))
        labels_empty = np.array([], dtype=int)
        
        with self.assertRaises(ValueError):
            cluster_quality_report(X_empty, labels_empty, self.centroids)

    def test_cluster_quality_summary_returns_string(self):
        """Test that cluster_quality_summary returns a formatted string."""
        result = cluster_quality_report(self.X, self.labels, self.centroids)
        summary = cluster_quality_summary(result)
        
        self.assertIsInstance(summary, str)
        self.assertIn("Cluster Quality Assessment", summary)
        self.assertIn("Silhouette Score", summary)
        self.assertIn("Davies-Bouldin", summary)

    def test_cluster_quality_summary_contains_metrics(self):
        """Test that cluster_quality_summary includes all metric values."""
        result = cluster_quality_report(self.X, self.labels, self.centroids)
        summary = cluster_quality_summary(result)
        
        # Check that numeric values appear in the summary
        self.assertIn(f"{result['silhouette_score']:.4f}", summary)
        self.assertIn(f"{result['davies_bouldin_index']:.4f}", summary)

    def test_well_separated_clusters_quality(self):
        """Test that well-separated clusters have good quality metrics."""
        result = cluster_quality_report(self.X, self.labels, self.centroids)
        
        # Well-separated clusters should have high silhouette score
        self.assertGreater(result['silhouette_score'], 0.5)
        
        # Well-separated clusters should have low Davies-Bouldin index
        self.assertLess(result['davies_bouldin_index'], 1.0)


if __name__ == "__main__":
    unittest.main()
