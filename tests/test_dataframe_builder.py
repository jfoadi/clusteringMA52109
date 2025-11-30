###
## cluster_maker - test file
## James Foadi - University of Bath
## November 2025
###

import unittest

import numpy as np
import pandas as pd

from cluster_maker.dataframe_builder import define_dataframe_structure, simulate_data
from cluster_maker.data_analyser import analyse_numeric_features


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
    def setUp(self):
        """Setup a sample DataFrame meeting all Task 3c criteria."""
        self.test_df = pd.DataFrame({
            'A_num': [1.0, 2.0, 3.0, 4.0, np.nan],        # Numeric, 1 missing value (NaN)
            'B_num': [10, 20, 30, 40, 50],                # Numeric
            'C_num': [-1.0, 0.0, 1.0, 2.0, 3.0],          # Numeric
            'D_cat': ['a', 'b', 'c', 'd', 'e'],           # Non-numeric
            'E_bool': [True, False, True, False, True]    # Non-numeric
        })
        # Expected results for numeric columns, based on 5 rows total.
        # A_num stats based on 4 non-missing values (1, 2, 3, 4)
        self.expected_A = {'mean': 2.5, 'missing': 1.0}
        self.expected_B = {'std': 15.8114, 'max': 50.0, 'missing': 0.0}
        self.expected_C = {'min': -1.0, 'missing': 0.0}
        
    def test_analysis_function_meets_task3c_requirements(self):
        """
        Tests analyse_numeric_features for statistical correctness, 
        missing value count, and non-numeric robustness.
        """
        
        # 1. Run the analysis function
        summary_df = analyse_numeric_features(self.test_df)
        
        # --- Assertions ---

        # A. Check dimensions and non-numeric column handling
        self.assertEqual(len(summary_df.index), 3, 
                         "Summary DataFrame should only contain 3 numeric columns.")
        self.assertNotIn('D_cat', summary_df.index, 
                         "Non-numeric (string) column D_cat should be ignored.")
        self.assertNotIn('E_bool', summary_df.index, 
                         "Non-numeric (boolean) column E_bool should be ignored.")

        # B. Check calculated values for the column with missing data (A_num)
        a_row = summary_df.loc['A_num']
        self.assertAlmostEqual(a_row['mean'], self.expected_A['mean'], places=3, 
                               msg="Mean for A_num is incorrect.")
        self.assertAlmostEqual(a_row['missing'], self.expected_A['missing'], 
                               msg="Missing count for A_num is incorrect.")

        # C. Check calculated values for other numeric columns (B_num and C_num)
        b_row = summary_df.loc['B_num']
        self.assertAlmostEqual(b_row['std'], self.expected_B['std'], places=3, 
                               msg="Std Dev for B_num is incorrect.")
        self.assertAlmostEqual(b_row['max'], self.expected_B['max'], 
                               msg="Max for B_num is incorrect.")
        
        c_row = summary_df.loc['C_num']
        self.assertAlmostEqual(c_row['min'], self.expected_C['min'], 
                               msg="Min for C_num is incorrect.")
        self.assertAlmostEqual(c_row['missing'], self.expected_C['missing'], 
                               msg="Missing count for C_num is incorrect.")


if __name__ == "__main__":
    unittest.main()