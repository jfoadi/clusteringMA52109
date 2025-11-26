###
## cluster_maker - test file
## James Foadi - University of Bath
## November 2025
###

import unittest

import numpy as np
import pandas as pd

from cluster_maker.dataframe_builder import define_dataframe_structure, simulate_data 
from cluster_maker.data_analyser import summarize_numeric_data

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
        
        
    def test_summarize_numeric_data_robustness(self):
    # --- 1. Setup: Create a DataFrame satisfying all requirements ---
        data = {
            # Numeric Column 1: Simple integers (mean=3.0, std=1.581, missing=0)
            'A_Int': [1, 2, 3, 4, 5],
            
            # Numeric Column 2: Floats with one missing value (mean=27.5, missing=1)
            'B_Float': [10.0, 20.0, 30.0, np.nan, 50.0],
            
            # Numeric Column 3: Another numeric column (mean=103.0, missing=0)
            'C_ID': [101, 102, 103, 104, 105], 
            
            # Non-Numeric Column: Must be ignored
            'D_Category': ['x', 'y', 'z', 'x', 'y']
        }
        input_df = pd.DataFrame(data)

        # --- 2. Execution ---
        summary_df = summarize_numeric_data(input_df)

        # --- 3. Assertions ---

        # A. Check shape and column names (3 rows for 3 numeric features, 5 statistics columns)
        self.assertEqual(summary_df.shape, (3, 5), 
                            "Summary DataFrame has incorrect shape (rows=features, columns=stats).")
        self.assertListEqual(
            list(summary_df.columns), 
            ['mean', 'std', 'min', 'max', 'missing_count'],
            "Summary DataFrame columns are incorrect."
        )

        # B. Check for non-numeric column exclusion (robustness check)
        self.assertNotIn('D_Category', summary_df.index, 
                            "Non-numeric column 'D_Category' was not excluded.")

        # C. Check specific statistics and missing value handling
        
        # Test for Column 'A_Int'
        self.assertAlmostEqual(summary_df.loc['A_Int', 'mean'], 3.0)
        self.assertEqual(summary_df.loc['A_Int', 'missing_count'], 0)

        # Test for Column 'B_Float' (Missing value check is the most critical)
        self.assertEqual(summary_df.loc['B_Float', 'missing_count'], 1, 
                            "Missing value count for 'B_Float' is incorrect.")
        
        # Test the mean calculation for 'B_Float' (mean of 4 non-NaN values: 110 / 4 = 27.5)
        self.assertAlmostEqual(summary_df.loc['B_Float', 'mean'], 27.5)
        self.assertAlmostEqual(summary_df.loc['B_Float', 'min'], 10.0)
        self.assertAlmostEqual(summary_df.loc['B_Float', 'max'], 50.0)
        
        # Test for Column 'C_ID'
        self.assertAlmostEqual(summary_df.loc['C_ID', 'mean'], 103.0) 


if __name__ == "__main__":
    unittest.main()