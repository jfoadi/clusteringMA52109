###
## cluster_maker  Interface and Export Test File
## Mock Practical MA52109 Task 5)
## November 2025
###

import unittest
import os
import tempfile
import shutil

import pandas as pd
import numpy as np

# Import functions to test
from cluster_maker.interface import run_clustering
from cluster_maker.data_exporter import export_to_csv, export_statistics_summary


class TestInterfaceAndExport(unittest.TestCase):
    """
    Test suite for high-level interface error handling and export functionality.
    """
    
    def setUp(self):
        """
        Set up test environment before each test.
        """
        print("Setting up test environment...")
        self.test_dir = tempfile.mkdtemp()
        print(f"Created temporary directory: {self.test_dir}")
        
        # Create a small test DataFrame for export tests
        self.test_stats_data = {
            'mean': [1.5, 2.5, 3.5],
            'std': [0.5, 0.8, 1.2],
            'min': [1.0, 2.0, 3.0],
            'max': [2.0, 3.0, 4.0],
            'missing': [0, 1, 0]
        }
        self.test_stats_df = pd.DataFrame(
            self.test_stats_data,
            index=['feature1', 'feature2', 'feature3']
        )
        print("Test DataFrame created successfully")
        print("-" * 50)
    
    def tearDown(self):
        """
        Clean up after each test.
        """
        print("Cleaning up test environment...")
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        print("Test environment cleaned successfully")
        print("=" * 60)
    # Task 5a)
    def test_interface_missing_file(self):
        """
        Test 5a: run_clustering with missing input file should raise clean error.
        """
        print("TEST 5a: Testing run_clustering with missing input file...")
        
        # Create a path to a non-existent file
        missing_file = os.path.join(self.test_dir, "non_existent_file.csv")
        print(f"Using non-existent file: {missing_file}")
        
        # This should raise a clean error, not a raw traceback
        with self.assertRaises(Exception) as context:
            run_clustering(
                input_path=missing_file,
                feature_cols=['x', 'y'],
                algorithm='kmeans',
                k=3
            )
        
        # Verify it's a controlled error (not a raw traceback)
        print("Checking error type and message...")
        self.assertIsNotNone(context.exception)
        error_message = str(context.exception)
        self.assertIsInstance(error_message, str)
        self.assertNotIn("Traceback", error_message)
        self.assertNotIn("File", error_message)  # Should not show raw file path in traceback
        
        print(" Missing file test passed: Clean error raised without raw traceback")
        print(f"   Error type: {type(context.exception).__name__}")
        print(f"   Error message: {error_message}")
    
    def test_interface_missing_columns(self):
        """
        Test 5a: run_clustering with missing feature columns should raise clean error.
        """
        print("TEST 5a: Testing run_clustering with missing feature columns...")
        
        # Create a temporary CSV file with some data but not the required columns
        test_csv_path = os.path.join(self.test_dir, "test_data.csv")
        test_data = {
            'col_a': [1, 2, 3, 4, 5],
            'col_b': [10, 20, 30, 40, 50]
        }
        test_df = pd.DataFrame(test_data)
        test_df.to_csv(test_csv_path, index=False)
        print(f"Created test CSV file: {test_csv_path}")
        print(f"Columns in test file: {list(test_df.columns)}")
        
        # Try to use columns that don't exist - should raise clean error
        with self.assertRaises(Exception) as context:
            run_clustering(
                input_path=test_csv_path,
                feature_cols=['nonexistent_col1', 'nonexistent_col2'],  # These don't exist
                algorithm='kmeans',
                k=3
            )
        
        # Verify it's a controlled error
        print("Checking error type and message...")
        self.assertIsNotNone(context.exception)
        error_message = str(context.exception)
        self.assertIsInstance(error_message, str)
        self.assertNotIn("Traceback", error_message)
        
        print(" Missing columns test passed: Clean error raised without raw traceback")
        print(f"   Error type: {type(context.exception).__name__}")
        print(f"   Error message: {error_message}")

    # Task 5b)
    def test_export_functions_valid_inputs(self):
        """
        Test 5b: Export functions should create output files with valid inputs.
        """
        print("TEST 5b: Testing export functions with valid inputs...")
        
        # Test export_to_csv with valid inputs
        csv_output_path = os.path.join(self.test_dir, "test_output.csv")
        print(f"Testing export_to_csv with output path: {csv_output_path}")
        
        # Create a simple DataFrame for testing
        test_df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        export_to_csv(test_df, csv_output_path)
        
        # Verify file was created
        self.assertTrue(os.path.exists(csv_output_path))
        print(" export_to_csv: File created successfully")
        
        # Verify file content can be read back
        read_back_df = pd.read_csv(csv_output_path)
        self.assertEqual(len(read_back_df), 3)
        print(" export_to_csv: File content is valid")
        
        # Test export_statistics_summary with valid inputs
        stats_csv_path = os.path.join(self.test_dir, "stats_output.csv")
        stats_text_path = os.path.join(self.test_dir, "stats_output.txt")
        print(f"Testing export_statistics_summary:")
        print(f"  CSV output: {stats_csv_path}")
        print(f"  Text output: {stats_text_path}")
        
        export_statistics_summary(self.test_stats_df, stats_csv_path, stats_text_path)
        
        # Verify both files were created
        self.assertTrue(os.path.exists(stats_csv_path))
        self.assertTrue(os.path.exists(stats_text_path))
        print(" export_statistics_summary: Both output files created")
        
        # Verify CSV file content
        stats_csv_df = pd.read_csv(stats_csv_path, index_col=0)
        self.assertEqual(stats_csv_df.shape, self.test_stats_df.shape)
        print(" export_statistics_summary: CSV file content is valid")
        
        # Verify text file content
        with open(stats_text_path, 'r') as f:
            text_content = f.read()
        self.assertGreater(len(text_content), 0)
        self.assertIn("COLUMN STATISTICS SUMMARY", text_content)
        print(" export_statistics_summary: Text file content is valid")
        
        print(" All export function tests passed: Files created successfully with valid inputs")
    
    def test_export_functions_invalid_paths(self):
        """
        Test 5b: Export functions should raise clear errors with invalid paths.
        """
        print("TEST 5b: Testing export functions with invalid paths...")
        
        # Test export_to_csv with invalid path
        invalid_csv_path = os.path.join(self.test_dir, "nonexistent_dir", "output.csv")
        print(f"Testing export_to_csv with invalid path: {invalid_csv_path}")
        
        test_df = pd.DataFrame({'x': [1, 2, 3]})
        
        with self.assertRaises(Exception) as context:
            export_to_csv(test_df, invalid_csv_path)
        
        # Verify it's a controlled error
        error_message = str(context.exception)
        self.assertNotIn("Traceback", error_message)
        self.assertIsInstance(error_message, str)
        print(" export_to_csv: Clean error raised for invalid path")
        print(f"   Error type: {type(context.exception).__name__}")
        print(f"   Error message: {error_message}")
        
        # Test export_statistics_summary with invalid path
        invalid_stats_csv = os.path.join(self.test_dir, "bad_dir", "stats.csv")
        invalid_stats_text = os.path.join(self.test_dir, "bad_dir", "stats.txt")
        print(f"Testing export_statistics_summary with invalid paths:")
        print(f"  Invalid CSV path: {invalid_stats_csv}")
        print(f"  Invalid text path: {invalid_stats_text}")
        
        with self.assertRaises(Exception) as context:
            export_statistics_summary(
                self.test_stats_df, 
                invalid_stats_csv, 
                invalid_stats_text
            )
        
        # Verify it's a controlled error
        error_message = str(context.exception)
        self.assertNotIn("Traceback", error_message)
        self.assertIsInstance(error_message, str)
        print(" export_statistics_summary: Clean error raised for invalid paths")
        print(f"   Error type: {type(context.exception).__name__}")
        print(f"   Error message: {error_message}")
    
    def test_interface_complete_workflow(self):
        """
        Additional test: Verify that run_clustering works correctly with valid inputs.
        """
        print("ADDITIONAL TEST: Testing complete workflow with valid inputs...")
        
        # Create a valid test CSV file
        valid_csv_path = os.path.join(self.test_dir, "valid_data.csv")
        print(f"Creating valid test CSV: {valid_csv_path}")
        
        # Create synthetic clustered data
        np.random.seed(42)
        data = {
            'feature_x': np.concatenate([
                np.random.normal(0, 0.5, 50),
                np.random.normal(3, 0.5, 50),
                np.random.normal(6, 0.5, 50)
            ]),
            'feature_y': np.concatenate([
                np.random.normal(0, 0.5, 50),
                np.random.normal(4, 0.5, 50),
                np.random.normal(8, 0.5, 50)
            ]),
            'id': range(150)
        }
        valid_df = pd.DataFrame(data)
        valid_df.to_csv(valid_csv_path, index=False)
        print(f"Created test data with {len(valid_df)} rows and columns: {list(valid_df.columns)}")
        
        # Test that run_clustering works with valid inputs
        output_path = os.path.join(self.test_dir, "clustered_output.csv")
        print(f"Running run_clustering with output path: {output_path}")
        
        try:
            result = run_clustering(
                input_path=valid_csv_path,
                feature_cols=['feature_x', 'feature_y'],
                algorithm='kmeans',
                k=3,
                output_path=output_path,
                random_state=42
            )
            
            # Verify results
            self.assertIn('data', result)
            self.assertIn('labels', result)
            self.assertIn('centroids', result)
            self.assertIn('metrics', result)
            self.assertTrue(os.path.exists(output_path))
            
            print(" Complete workflow test passed:")
            print(f"   - Result contains all expected keys")
            print(f"   - Output file created: {output_path}")
            print(f"   - Metrics computed: {list(result['metrics'].keys())}")
            
        except Exception as e:
            self.fail(f"run_clustering failed with valid inputs: {e}")


if __name__ == "__main__":
    print("STARTING TASK 5 TESTS: Interface and Export Functionality")
    print("=" * 70)
    unittest.main(verbosity=2)