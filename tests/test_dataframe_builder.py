###
## cluster_maker - test file
## James Foadi - University of Bath
## November 2025
###

import unittest

import numpy as np
import pandas as pd

from cluster_maker.dataframe_builder import define_dataframe_structure, simulate_data
from cluster_maker.data_analyser import calculate_column_statistics # Added in import for task 3c)

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


    def test_calculate_column_statistics_comprehensive(self):
        """
        Test the calculate_column_statistics function with a comprehensive dataset
        containing numeric columns, non-numeric columns, and missing values.
        """
        print(" Starting comprehensive test for calculate_column_statistics...")
        
        # Create test DataFrame with all required elements using only allowed libraries
        test_data = {
            'numeric_col1': [1.0, 2.0, 3.0, 4.0, 5.0, None],  # 6 values, 1 missing (using None instead of np.nan)
            'numeric_col2': [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],  # 6 values, 0 missing
            'numeric_col3': [100.0, None, 300.0, 400.0, 500.0, 600.0],  # 6 values, 1 missing (using None)
            'string_col': ['A', 'B', 'C', 'D', 'E', 'F'],  # Non-numeric column
            'category_col': ['X', 'Y', 'X', 'Y', 'X', 'Y']  # Another non-numeric column
        }
        
        test_df = pd.DataFrame(test_data)
        print(" Created test DataFrame with:")
        print(f"   - 3 numeric columns: ['numeric_col1', 'numeric_col2', 'numeric_col3']")
        print(f"   - 2 non-numeric columns: ['string_col', 'category_col']")
        print(f"   - Missing values: 2 total across numeric columns")
        print(f"   - DataFrame shape: {test_df.shape}")
        
        
        # Execute the function
        print(" Running calculate_column_statistics...")
        result = calculate_column_statistics(test_df)
        
        # Verify the results
        print(" Verifying test results...")
        
        # Test 1: Result should be a DataFrame
        self.assertIsInstance(result, pd.DataFrame)
        print(" Result is a pandas DataFrame")
        
        # Test 2: Should have 3 rows (one for each numeric column)
        self.assertEqual(result.shape[0], 3)
        print("Correct number of rows (3 numeric columns)")
        
        # Test 3: Should have 5 columns (mean, std, min, max, missing)
        self.assertEqual(result.shape[1], 5)
        print("Correct number of columns (5 statistics)")
        
        # Test 4: Should have correct column names
        expected_columns = ['mean', 'std', 'min', 'max', 'missing']
        self.assertListEqual(list(result.columns), expected_columns)
        print("Correct column names")
        
        # Test 5: Should have correct index (numeric column names)
        expected_index = ['numeric_col1', 'numeric_col2', 'numeric_col3']
        self.assertListEqual(list(result.index), expected_index)
        print("Correct index (only numeric columns included)")
        
        # Test 6: Check specific values for numeric_col1
        col1_stats = result.loc['numeric_col1']
        expected_mean_col1 = (1.0 + 2.0 + 3.0 + 4.0 + 5.0) / 5  # Mean of non-None values
        self.assertAlmostEqual(col1_stats['mean'], expected_mean_col1, places=6)
        self.assertGreater(col1_stats['std'], 0)  # Std should be positive
        self.assertEqual(col1_stats['min'], 1.0)
        self.assertEqual(col1_stats['max'], 5.0)
        self.assertEqual(col1_stats['missing'], 1)
        print(" numeric_col1 statistics are correct")
        
        # Test 7: Check specific values for numeric_col2 (no missing values)
        col2_stats = result.loc['numeric_col2']
        expected_mean_col2 = (10.0 + 20.0 + 30.0 + 40.0 + 50.0 + 60.0) / 6
        self.assertAlmostEqual(col2_stats['mean'], expected_mean_col2, places=6)
        self.assertGreater(col2_stats['std'], 0)
        self.assertEqual(col2_stats['min'], 10.0)
        self.assertEqual(col2_stats['max'], 60.0)
        self.assertEqual(col2_stats['missing'], 0)
        print(" numeric_col2 statistics are correct")
        
        # Test 8: Check specific values for numeric_col3 (with missing value)
        col3_stats = result.loc['numeric_col3']
        expected_mean_col3 = (100.0 + 300.0 + 400.0 + 500.0 + 600.0) / 5  # Mean of non-None values
        self.assertAlmostEqual(col3_stats['mean'], expected_mean_col3, places=6)
        self.assertGreater(col3_stats['std'], 0)
        self.assertEqual(col3_stats['min'], 100.0)
        self.assertEqual(col3_stats['max'], 600.0)
        self.assertEqual(col3_stats['missing'], 1)
        print(" numeric_col3 statistics are correct")
        
        # Test 9: Verify non-numeric columns are excluded
        self.assertNotIn('string_col', result.index)
        self.assertNotIn('category_col', result.index)
        print("Non-numeric columns correctly excluded")
        
        # Test 10: Verify all statistics are computed
        for col in expected_index:
            for stat in expected_columns:
                self.assertIn(stat, result.columns)
                self.assertIsNotNone(result.loc[col, stat])
        print(" All statistics computed for all numeric columns")
        
        print(" All tests passed successfully!")
        print(" Final result summary:")
        print(f"   - Rows analyzed: {result.shape[0]}")
        print(f"   - Statistics computed: {result.shape[1]}")
        print(f"   - Missing values correctly counted")
        print(f"   - Non-numeric columns properly excluded")
        print("Task 3c: Comprehensive test function completed successfully!")
        print("-" * 60)
    
if __name__ == "__main__":
    unittest.main()