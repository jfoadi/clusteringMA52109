###
## cluster_maker - test file
## James Foadi - University of Bath
## November 2025
###

import unittest

import numpy as np
import pandas as pd

from cluster_maker.dataframe_builder import define_dataframe_structure, simulate_data
from cluster_maker.data_analyser import get_numeric_column_summary


class TestDataFrameBuilder(unittest.TestCase):
    def test_define_dataframe_structure_basic(self):
        column_specs = [
            {"name": "x", "reps": [0.0, 1.0, 2.0]},
            {"name": "y", "reps": [10.0, 11.0, 12.0]},
        ]
        seed_df = define_dataframe_structure(column_specs)
        self.assertEqual(seed_df.shape, (3, 2))
        self.assertListEqual(list(seed_df.columns), ["x", "y"])
        self.assertTrue(np.allclose(seed_df["x"].values, [0.0, 1.0, 2.0]))

    def test_simulate_data_shape(self):
        column_specs = [
            {"name": "x", "reps": [0.0, 5.0]},
            {"name": "y", "reps": [2.0, 4.0]},
        ]
        seed_df = define_dataframe_structure(column_specs)
        data = simulate_data(seed_df, n_points=100, random_state=1)
        self.assertEqual(data.shape[0], 100)
        self.assertIn("true_cluster", data.columns)







class TestAnalysisFunctions(unittest.TestCase):
    """Tests for the data analysis functions, including the new summary function."""

    def test_numeric_summary_robustness(self):
        """
        Checks that get_numeric_column_summary:
        1. Ignores non-numeric columns (Category).
        2. Correctly handles missing values (Col_A).
        3. Returns the correct summary statistics.
        """
        print("\n--- Running Test: test_numeric_summary_robustness ---")
        
        # 1. Setup the complex test DataFrame
        data = pd.DataFrame({
            "Col_A": [np.nan, 2.0, 3.0, 4.0, 5.0],  # Has NaN
            "Col_B": [10, 10, 10, 10, 10],          # Simple integers
            "Col_C": [1.0, 2.0, 3.0, 4.0, 5.0],     # Normal floats
            "Category": ["cat1", "cat2", "cat3", "cat4", "cat5"], # Non-numeric
            "Col_D": pd.to_datetime(["2023-01-01"] * 5) # Non-numeric date type
        })

        # 2. Define the expected output based on manual calculation
        expected_stats = pd.DataFrame({
            # Col_A (NaN at [0], data: 2, 3, 4, 5)
            "Col_A": {
                "mean": 3.5,
                "std": 1.2909944,  # np.sqrt(1.666667 * 3)
                "min": 2.0,
                "max": 5.0,
                "missing_values": 1,
            },
            # Col_B (data: 10, 10, 10, 10, 10)
            "Col_B": {
                "mean": 10.0,
                "std": 0.0,
                "min": 10.0,
                "max": 10.0,
                "missing_values": 0,
            },
            # Col_C (data: 1, 2, 3, 4, 5)
            "Col_C": {
                "mean": 3.0,
                "std": 1.5811388,  # np.std(np.array([1, 2, 3, 4, 5]), ddof=1)
                "min": 1.0,
                "max": 5.0,
                "missing_values": 0,
            }
        })
        expected_stats.index.name = "statistic"

        # 3. Call the function
        summary_df = get_numeric_column_summary(data)

        # 4. Assertions

        # A. Check shape and columns (Robustness)
        self.assertEqual(summary_df.shape, (5, 3), 
                         "The final summary shape is incorrect. Should be (5 stats, 3 features).")
        self.assertListEqual(list(summary_df.columns), ["Col_A", "Col_B", "Col_C"],
                             "Non-numeric columns were incorrectly included.")

        # B. Check contents (Correctness)
        pd.testing.assert_frame_equal(
            summary_df.sort_index(axis=1),
            expected_stats.sort_index(axis=1),
            check_dtype=False, # Allow for slight float/int differences
            check_exact=False, # Use tolerance for floats
            atol=1e-6 # Absolute tolerance for floating point comparisons
        )
        print("  -> All statistical assertions passed successfully.")
        print("--- Test Complete: test_numeric_summary_robustness ---")


if __name__ == "__main__":
    unittest.main()




if __name__ == "__main__":
    unittest.main()