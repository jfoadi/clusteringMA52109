import unittest
import numpy as np
from cluster_maker.evaluation import calculate_bgss

class TestClusterEvaluation(unittest.TestCase):

    def test_bgss_numerical_correctness(self):
        """
        Verifies BGSS calculation on a simple, hand-calculated dataset.
        
        Data: X = [[1, 1], [2, 2], [10, 10], [11, 11]]
        Labels: [0, 0, 1, 1]
        
        Overall Mean (x_bar): [6.0, 6.0]
        
        Cluster 0 Mean (x_bar_0): [1.5, 1.5], n_0 = 2
        Cluster 1 Mean (x_bar_1): [10.5, 10.5], n_1 = 2
        
        BGSS = n_0 * ||x_bar_0 - x_bar||^2 + n_1 * ||x_bar_1 - x_bar||^2
        
        Term 0: 2 * ( (1.5 - 6.0)^2 + (1.5 - 6.0)^2 )
              = 2 * ( (-4.5)^2 * 2 ) = 4 * 20.25 = 81.0
              
        Term 1: 2 * ( (10.5 - 6.0)^2 + (10.5 - 6.0)^2 )
              = 2 * ( (4.5)^2 * 2 ) = 4 * 20.25 = 81.0
              
        Expected BGSS = 81.0 + 81.0 = 162.0
        """
        X = np.array([[1, 1], [2, 2], [10, 10], [11, 11]])
        labels = np.array([0, 0, 1, 1])
        
        expected_bgss = 162.0
        calculated_bgss = calculate_bgss(X, labels)
        
        self.assertIsInstance(calculated_bgss, float)
        self.assertAlmostEqual(calculated_bgss, expected_bgss, places=6)
        
    def test_bgss_zero_separation(self):
        """Checks BGSS is zero when all points belong to one cluster."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        labels = np.array([0, 0, 0])
        self.assertAlmostEqual(calculate_bgss(X, labels), 0.0)
        
    def test_bgss_error_on_mismatch(self):
        """Checks for ValueError if data and labels shapes mismatch."""
        X = np.array([[1, 2], [3, 4]])
        labels = np.array([0]) # Missing a label
        with self.assertRaises(ValueError):
            calculate_bgss(X, labels)

# if __name__ == "__main__":
#     unittest.main()