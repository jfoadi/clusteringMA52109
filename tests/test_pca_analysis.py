###
## cluster_maker  PCA Analysis Test File
## Mock Practical MA52109 Task 6)
## November 2025
###

import unittest
import numpy as np
import pandas as pd

from cluster_maker.pca_analysis import (
    compute_pca,
    explained_variance_analysis,
    plot_pca_variance,
    pca_clustering_workflow
)


class TestPCAAnalysis(unittest.TestCase):
    """
    Test suite for PCA analysis extension functionality.
    """
    
    def setUp(self):
        """
        Set up test data for PCA analysis.
        """
        print("Setting up test data for PCA analysis...")
        np.random.seed(42)
        
        # Create synthetic data with clear principal components
        n_samples = 100
        n_features = 5
        
        # Create correlated data to ensure meaningful PCA
        base_component = np.random.normal(0, 2, n_samples)
        self.X = np.column_stack([
            base_component + np.random.normal(0, 0.1, n_samples),  # Strong first component
            base_component * 0.5 + np.random.normal(0, 0.1, n_samples),  # Correlated second
            np.random.normal(0, 1, n_samples),  # Independent third
            np.random.normal(0, 1, n_samples),  # Independent fourth  
            np.random.normal(0, 1, n_samples)   # Independent fifth
        ])
        
        print(f"Created test data with {n_samples} samples and {n_features} features")
        print("Test data setup completed")
        print("-" * 50)
    
    def test_compute_pca_basic_functionality(self):
        """
        Test basic PCA computation with valid input data.
        """
        print("TEST: Basic PCA computation functionality...")
        
        # Test with standardization
        X_pca, pca_model, scaler = compute_pca(self.X, n_components=3, standardize=True)
        
        # Verify output shapes and types
        self.assertEqual(X_pca.shape[0], self.X.shape[0])
        self.assertEqual(X_pca.shape[1], 3)
        self.assertIsNotNone(pca_model)
        self.assertIsNotNone(scaler)
        
        print(" Basic PCA computation test passed")
    
    def test_compute_pca_numerical_stability(self):
        """
        Test numerical stability of PCA computation.
        """
        print("TEST: PCA numerical stability...")
        
        # Test with different numbers of components
        for n_comp in [2, 3, None]:
            with self.subTest(n_components=n_comp):
                X_pca, pca_model, _ = compute_pca(self.X, n_components=n_comp)
                
                # Check that explained variance ratios are reasonable
                explained_variance = pca_model.explained_variance_ratio_
                
                # All variances should be between 0 and 1
                self.assertTrue(np.all(explained_variance >= 0))
                self.assertTrue(np.all(explained_variance <= 1))
                
                # Sum of variances should be <= 1 (might be <1 if n_components < n_features)
                if n_comp is None or n_comp == self.X.shape[1]:
                    self.assertAlmostEqual(np.sum(explained_variance), 1.0, places=6)
                else:
                    self.assertTrue(np.sum(explained_variance) <= 1.0)
        
        print(" PCA numerical stability test passed")
    
    def test_explained_variance_analysis(self):
        """
        Test explained variance analysis functionality.
        """
        print("TEST: Explained variance analysis...")
        
        X_pca, pca_model, _ = compute_pca(self.X, n_components=3)
        variance_info = explained_variance_analysis(pca_model)
        
        # Check that all required keys are present
        required_keys = [
            'explained_variance_ratio', 
            'cumulative_variance_ratio',
            'n_components_95', 
            'n_components_90',
            'total_variance_explained'
        ]
        
        for key in required_keys:
            self.assertIn(key, variance_info)
        
        # Check that cumulative variance is monotonically increasing
        cumulative_var = variance_info['cumulative_variance_ratio']
        self.assertTrue(np.all(np.diff(cumulative_var) >= 0))
        
        # Check that component counts are reasonable
        self.assertGreaterEqual(variance_info['n_components_90'], 1)
        self.assertLessEqual(variance_info['n_components_90'], 3)
        self.assertGreaterEqual(variance_info['n_components_95'], 1)
        self.assertLessEqual(variance_info['n_components_95'], 3)
        
        print(" Explained variance analysis test passed")
    
    def test_pca_clustering_workflow_integration(self):
        """
        Test complete PCA-clustering workflow integration.
        """
        print("TEST: PCA-clustering workflow integration...")
        
        result = pca_clustering_workflow(
            self.X,
            n_pca_components=2,
            clustering_algorithm="kmeans",
            k=3,
            random_state=42
        )
        
        # Verify all result components are present
        self.assertIn('X_original', result)
        self.assertIn('X_pca', result)
        self.assertIn('pca_model', result)
        self.assertIn('variance_info', result)
        self.assertIn('clustering_result', result)
        self.assertIn('fig_variance', result)
        
        # Verify clustering results
        clustering_result = result['clustering_result']
        self.assertIn('labels', clustering_result)
        self.assertIn('centroids', clustering_result)
        self.assertIn('metrics', clustering_result)
        
        # Verify we have the expected number of clusters
        unique_labels = np.unique(clustering_result['labels'])
        self.assertEqual(len(unique_labels), 3)
        
        print(" PCA-clustering workflow integration test passed")
    
    def test_pca_error_handling(self):
        """
        Test PCA error handling with invalid inputs.
        """
        print("TEST: PCA error handling...")
        
        # Test with 1D array
        with self.assertRaises(ValueError):
            compute_pca(np.array([1, 2, 3]))
        
        # Test with insufficient samples
        with self.assertRaises(ValueError):
            compute_pca(np.array([[1, 2]]))  # Only 1 sample
        
        # Test with insufficient features  
        with self.assertRaises(ValueError):
            compute_pca(np.array([[1], [2]]))  # Only 1 feature
        
        print(" PCA error handling test passed")


if __name__ == "__main__":
    print("STARTING TASK 6 TESTS: PCA Analysis Extension")
    print("=" * 70)
    unittest.main(verbosity=2)
