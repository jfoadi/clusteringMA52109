###
## cluster_maker - test file
## James Foadi - University of Bath
## November 2025
###

import unittest

import numpy as np
import pandas as pd

from cluster_maker.dataframe_builder import define_dataframe_structure, simulate_data
from cluster_maker.data_analyser import summarise_numeric_data # <-- NEW IMPORT


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

    def test_summarise_numeric_data(self):
        """
        Tests the new analysis function (summarise_numeric_data) for correctness, 
        handling of non-numeric columns, and accurate missing value counts.
        """
        # 1. Setup: Create a DataFrame satisfying the requirements:
        #    - at least 3 numeric columns
        #    - at least 1 non-numeric column
        #    - at least 1 missing value
        data = {
            'A_numeric': [1.0, 2.0, 3.0, 4.0, np.nan],       # N=4, 1 missing
            'B_numeric': [10, 20, 30, 40, 50],               # N=5, 0 missing
            'C_numeric': [0.1, 0.2, 0.3, 0.4, 0.5],          # N=5, 0 missing
            'D_non_numeric': ['a', 'b', 'c', 'd', 'e'],      # Non-numeric (must be ignored)
        }
        df = pd.DataFrame(data)

        # 2. Execution
        summary_df = summarise_numeric_data(df)

        # 3. Assertions
        
        # Check shape: 5 statistics (mean, std, min, max, count_missing) x 3 numeric columns
        self.assertEqual(summary_df.shape, (5, 3), "Summary DataFrame shape is incorrect.")
        
        # Check columns: Ensure only numeric columns are present [cite: 15]
        self.assertListEqual(
            list(summary_df.columns),
            ['A_numeric', 'B_numeric', 'C_numeric'],
            "Summary DataFrame columns are incorrect (non-numeric not ignored)."
        )

        # Check index names
        self.assertListEqual(
            list(summary_df.index),
            ['mean', 'std', 'min', 'max', 'count_missing'],
            "Summary DataFrame index names are incorrect."
        )

        # Check for correct calculation of missing count and mean for 'A_numeric'
        # N=5, 1 missing. Values sum to 10.0. Mean = 10.0/4 = 2.5
        self.assertAlmostEqual(summary_df.loc['mean', 'A_numeric'], 2.5) 
        self.assertEqual(summary_df.loc['count_missing', 'A_numeric'], 1) # [cite: 14]

        # Check for correct calculation for 'B_numeric' (no missing values)
        self.assertAlmostEqual(summary_df.loc['mean', 'B_numeric'], 30.0)
        self.assertEqual(summary_df.loc['count_missing', 'B_numeric'], 0) # [cite: 14]


if __name__ == "__main__":
    unittest.main()