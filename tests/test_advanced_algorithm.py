import unittest
import numpy as np
from cluster_maker.algorithms import sklearn_agglomerative

class TestHierarchicalClustering(unittest.TestCase):
    def test_agglomerative_basic_run(self):
        """
        Test that the new hierarchical clustering function runs without error
        and produces the expected output shapes.
        """
        # Create a simple dataset: 10 points, 2 features
        X = np.random.rand(10, 2)
        k = 3
        
        labels, centroids = sklearn_agglomerative(X, k=k)
        
        # Check shapes
        self.assertEqual(labels.shape, (10,))
        self.assertEqual(centroids.shape, (k, 2))
        
        # Check that we actually have 3 unique clusters (0, 1, 2)
        # Note: In rare cases with very tiny N, a cluster might be empty, 
        # but with N=10 and k=3, Agglomerative guarantees 3 clusters.
        unique_labels = np.unique(labels)
        self.assertEqual(len(unique_labels), k)
    
    def test_agglomerative_centroid_calculation(self):
        """
        Test that our manual centroid calculation is correct.
        """
        # Create two distinct points that will definitely be separate clusters if k=2
        # Point A at (0,0), Point B at (10,10)
        # We add a third point close to A to ensure A gets a cluster.
        X = np.array([
            [0.0, 0.0],
            [0.1, 0.1],
            [10.0, 10.0]
        ])
        
        labels, centroids = sklearn_agglomerative(X, k=2)
        
        # We expect 2 clusters. One should contain the points near (0,0),
        # the other should contain (10,10).
        
        # Find the label for the cluster containing (10,10) (index 2)
        label_far = labels[2]
        centroid_far = centroids[label_far]
        
        # The centroid of a single point is the point itself
        self.assertTrue(np.allclose(centroid_far, [10.0, 10.0]))

    def test_agglomerative_determinism(self):
        """
        Test that the algorithm is deterministic (unlike K-Means).
        """
        X = np.random.rand(20, 3)
        
        labels1, _ = sklearn_agglomerative(X, k=3)
        labels2, _ = sklearn_agglomerative(X, k=3)
        
        # Should be identical every time
        np.testing.assert_array_equal(labels1, labels2)

if __name__ == '__main__':
    unittest.main()