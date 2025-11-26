###
## cluster_maker - test file
## James Foadi - University of Bath
## November 2025
###

import unittest

import numpy as np
import pandas as pd

from cluster_maker.dataframe_builder import define_dataframe_structure, simulate_data


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


if __name__ == "__main__":
    unittest.main()

# --- Added code for task 3 ---

from cluster_maker.data_analyser import calculate_descriptive_statistics

class TestDataAnalyser(unittest.TestCase):
    def test_calculate_stats_mixed_types(self):
        """
        Test that descriptive stats are calculated correctly, 
        robust to non-numeric data, and handle missing values.
        """
        # Create a dummy dataframe with mixed types
        df = pd.DataFrame({
            'A': [1.0, 2.0, 3.0],          # Simple numeric
            'B': [10.0, np.nan, 30.0],     # Numeric with missing value
            'C': ['cat', 'dog', 'mouse']   # Non-numeric (should be ignored)
        })
        
        # Run the function
        stats = calculate_descriptive_statistics(df)
        
        # CHECK 1: Robustness (Text column 'C' should NOT be in the results)
        self.assertNotIn('C', stats.index)
        self.assertIn('A', stats.index)
        self.assertIn('B', stats.index)
        
        # CHECK 2: Correctness (Mean of 1, 2, 3 is 2.0)
        self.assertEqual(stats.loc['A', 'mean'], 2.0)
        
        # CHECK 3: Missing Values (Column 'B' has 1 NaN)
        self.assertEqual(stats.loc['B', 'missing_values'], 1.0)
