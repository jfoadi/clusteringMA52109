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



def test_calculate_comprehensive_statistics(self):
    """Test the comprehensive statistics function with mixed data types and missing values."""
    # Create test DataFrame with mixed types and missing values
    test_data = pd.DataFrame({
        'numeric1': [1.0, 2.0, 3.0, 4.0, 5.0],
        'numeric2': [10.5, 20.5, None, 40.5, 50.5],  # Contains missing value
        'numeric3': [100, 200, 300, 400, 500],
        'text_col': ['A', 'B', 'C', 'D', 'E'],  # Non-numeric column
        'category_col': ['X', 'Y', 'X', 'Y', 'X']  # Another non-numeric column
    })
    
    # Calculate comprehensive statistics
    from cluster_maker.data_analyser import calculate_comprehensive_statistics
    result = calculate_comprehensive_statistics(test_data)
    
    # Verify the result structure
    self.assertEqual(result.shape[1], 5)  # Should have 5 statistics columns
    expected_columns = ['mean', 'std', 'min', 'max', 'missing_count']
    self.assertListEqual(list(result.columns), expected_columns)
    
    # Verify only numeric columns are included
    self.assertIn('numeric1', result.index)
    self.assertIn('numeric2', result.index)
    self.assertIn('numeric3', result.index)
    self.assertNotIn('text_col', result.index)
    self.assertNotIn('category_col', result.index)
    
    # Verify specific calculations
    self.assertAlmostEqual(result.loc['numeric1', 'mean'], 3.0)
    self.assertAlmostEqual(result.loc['numeric1', 'min'], 1.0)
    self.assertAlmostEqual(result.loc['numeric1', 'max'], 5.0)
    self.assertEqual(result.loc['numeric1', 'missing_count'], 0)
    
    # Check missing value handling
    self.assertEqual(result.loc['numeric2', 'missing_count'], 1)
    
    # Verify numeric3 (integer column handled correctly)
    self.assertAlmostEqual(result.loc['numeric3', 'mean'], 300.0)    