###
## cluster_maker - test file
## James Foadi - University of Bath
## November 2025
###

import unittest

import numpy as np
import pandas as pd

from cluster_maker.dataframe_builder import define_dataframe_structure, simulate_data
from cluster_maker.data_analyser import numeric_summary


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


class TestNumericSummary(unittest.TestCase):
    def test_numeric_summary_basic(self):
        """Test numeric_summary with multiple numeric and non-numeric columns."""
        # Create test DataFrame with:
        # - 3 numeric columns
        # - 1 non-numeric column
        # - 1 missing value
        data = pd.DataFrame({
            'col1': [1.0, 2.0, 3.0, np.nan, 5.0],
            'col2': [10.0, 20.0, 30.0, 40.0, 50.0],
            'col3': [100.0, 200.0, 300.0, 400.0, 500.0],
            'text_col': ['a', 'b', 'c', 'd', 'e'],
        })

        summary = numeric_summary(data)

        # Check shape: 3 numeric columns, 5 statistics
        self.assertEqual(summary.shape[0], 3)
        self.assertEqual(summary.shape[1], 5)

        # Check column names
        expected_stats = {'mean', 'std', 'min', 'max', 'missing_count'}
        self.assertEqual(set(summary.columns), expected_stats)

        # Check that non-numeric column is not included
        self.assertNotIn('text_col', summary.index)

        # Check specific values for col1 (has 1 missing value)
        self.assertEqual(summary.loc['col1', 'missing_count'], 1)
        self.assertEqual(summary.loc['col1', 'mean'], 2.75)  # (1+2+3+5)/4
        self.assertEqual(summary.loc['col1', 'min'], 1.0)
        self.assertEqual(summary.loc['col1', 'max'], 5.0)

        # Check col2 (no missing values)
        self.assertEqual(summary.loc['col2', 'missing_count'], 0)
        self.assertEqual(summary.loc['col2', 'mean'], 30.0)
        self.assertEqual(summary.loc['col2', 'min'], 10.0)
        self.assertEqual(summary.loc['col2', 'max'], 50.0)


if __name__ == "__main__":
    unittest.main()