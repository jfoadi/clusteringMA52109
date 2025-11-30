import unittest
import numpy as np
from cluster_maker import cluster_stability_score


class TestClusterStability(unittest.TestCase):

    def test_stability_returns_valid_value(self):
        """
        Basic test: stability score should be between 0 and 1,
        and the function should run without errors.
        """
        X = np.array([
            [1, 2],
            [3, 4],
            [5.0, 5.2],
            [5.1, 5.3],
        ])

        score = cluster_stability_score(X, k=2, n_runs=5, noise_scale=0.01)

        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        
        
    def test_stability_low_noise(self):
        """
        Stability should improve when noise is reduced.
        """

        rng = np.random.RandomState(0)

        # Asymmetric, differently-shaped clusters
        cluster1 = rng.normal(loc=[0, 0], scale=[1.5, 0.1], size=(200, 2))
        cluster2 = rng.normal(loc=[10, 8], scale=[0.1, 0.1], size=(200, 2))
        X = np.vstack([cluster1, cluster2])

        high_noise_score = cluster_stability_score(
            X,
            k=2,
            n_runs=15,
            noise_scale=3.0,
            random_state=0
        )

        low_noise_score = cluster_stability_score(
            X,
            k=2,
            n_runs=15,
            noise_scale=0.01,
            random_state=1
        )

        print("HIGH:", high_noise_score)
        print("LOW :", low_noise_score)

        self.assertGreater(low_noise_score, high_noise_score)



if __name__ == "__main__":
    unittest.main()
