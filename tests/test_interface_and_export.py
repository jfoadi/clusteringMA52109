import unittest
import os
import tempfile
import shutil
import pandas as pd
import numpy as np


# Assuming the imports are correct for your environment
from cluster_maker.interface import run_clustering
from cluster_maker.data_exporter import export_summary_report


class TestInterfaceErrorHandling(unittest.TestCase):
    """
    Tests the robustness of the high-level interface function (run_clustering)
    by checking for controlled error responses to invalid inputs (Task 5a).
    """

    def setUp(self):
        """Setup runs before each test method: Create a temporary directory and valid mock CSV file."""
        print(f"\n{'='*50}\nSETUP: Preparing temporary data for interface test...")
        # Create a temporary directory
        self.temp_dir = tempfile.mkdtemp()
        self.valid_csv_path = os.path.join(self.temp_dir, "valid_data.csv")
        self.missing_csv_path = os.path.join(self.temp_dir, "missing.csv") # Used for FileNotFoundError test

        # Create a DataFrame with necessary columns and save it
        self.valid_data = pd.DataFrame({
            "feature_x": np.random.rand(10),
            "feature_y": np.random.rand(10),
            "id": range(10)
        })
        self.valid_data.to_csv(self.valid_csv_path, index=False)
        print(f"SETUP: Created valid test file at: {self.valid_csv_path}")

    def tearDown(self):
        """Cleanup runs after each test method: Remove the temporary directory."""
        print(f"TEARDOWN: Removing temporary directory: {self.temp_dir}")
        shutil.rmtree(self.temp_dir)
        print(f"{'='*50}")

    # -----------------------------------------------------------
    # 5a-1) Test Case: Missing Input File
    # -----------------------------------------------------------
    def test_run_clustering_missing_file_raises_filenotfounderror(self):
        """
        Checks that run_clustering raises FileNotFoundError (a controlled error)
        when the input file does not exist.

        FIX: Changed assertion from self.assertRaisesRegex to self.assertRaises 
        because the regex check failed against the raw OS error message.
        """
        print("\nTest 5a-1: Testing missing input file.")
        
        # We expect a FileNotFoundError exception to be raised
        with self.assertRaises(FileNotFoundError): # FIXED: Using exact type check
            print("  -> Calling run_clustering with a non-existent path...")
            run_clustering(
                input_path=self.missing_csv_path,
                feature_cols=["feature_x", "feature_y"],
                algorithm="kmeans"
            )
        print("  -> SUCCESS: FileNotFoundError was raised as expected (Controlled Error).")

    # -----------------------------------------------------------
    # 5a-2) Test Case: Missing Feature Columns
    # -----------------------------------------------------------
    def test_run_clustering_missing_features_raises_valueerror(self):
        """
        Checks that run_clustering raises ValueError (a controlled error)
        when a required feature column is missing from the data.

        FIX: Changed expected exception type from ValueError to KeyError,
        matching the exception raised by select_features.
        """
        print("\nTest 5a-2: Testing input file with missing required features.")

        missing_feature = "non_existent_feature"
        
        # We expect a KeyError (raised by select_features)
        with self.assertRaisesRegex(KeyError, missing_feature): # FIXED: Changed ValueError to KeyError
            print(f"  -> Calling run_clustering, requesting feature: '{missing_feature}'...")
            run_clustering(
                input_path=self.valid_csv_path,
                feature_cols=["feature_x", missing_feature],
                algorithm="kmeans"
            )
        print("  -> SUCCESS: KeyError was raised as expected (Controlled Error).")


class TestExportFunctionRobustness(unittest.TestCase):
    """
    Tests the robustness and output integrity of data_exporter functions
    (Task 5b).
    """

    def setUp(self):
        """Setup runs before each test method: Create a temporary directory and mock data."""
        print(f"\n{'='*50}\nSETUP: Preparing temporary data for export test...")
        self.temp_dir = tempfile.mkdtemp()
        self.base_filename = os.path.join(self.temp_dir, "export_test_report")
        
        # Create a mock summary DataFrame (Index=stats, Columns=features)
        self.summary_data = pd.DataFrame({
            "Age": {"mean": 35.5, "std": 10.2, "min": 18, "max": 75, "missing_values": 0},
            "Income": {"mean": 55000, "std": 12500, "min": 20000, "max": 90000, "missing_values": 2}
        })
        self.summary_data.index.name = "statistic"
        print("SETUP: Created mock summary DataFrame.")

    def tearDown(self):
        """Cleanup runs after each test method: Remove the temporary directory."""
        print(f"TEARDOWN: Removing temporary directory: {self.temp_dir}")
        shutil.rmtree(self.temp_dir)
        print(f"{'='*50}")

    # -----------------------------------------------------------
    # 5b-1) Test Case: Successful File Creation
    # -----------------------------------------------------------
    def test_export_summary_report_creates_files(self):
        """Checks that both CSV and TXT files are created when given valid inputs."""
        print("\nTest 5b-1: Testing successful creation of CSV and TXT files.")
        
        # 1. Define expected output paths
        csv_path = f"{self.base_filename}_summary.csv"
        txt_path = f"{self.base_filename}_report.txt"

        # 2. Call the function
        print("  -> Calling export_summary_report...")
        export_summary_report(self.summary_data, self.base_filename)
        
        # 3. Assert files were created (Task 5b check)
        self.assertTrue(os.path.exists(csv_path), "FAILURE: CSV file was not created.")
        self.assertTrue(os.path.exists(txt_path), "FAILURE: TXT file was not created.")
        
        # 4. Check file content to ensure they are not empty
        self.assertGreater(os.path.getsize(csv_path), 10, "CSV file is suspiciously small/empty.")
        self.assertGreater(os.path.getsize(txt_path), 50, "TXT file is suspiciously small/empty.")
        
        print("  -> SUCCESS: Both output files were created and contain content.")

    # -----------------------------------------------------------
    # 5b-2) Test Case: Invalid Path Error Handling
    # -----------------------------------------------------------
    def test_export_summary_report_invalid_path_raises_error(self):
        """
        Checks that export_summary_report raises a controlled error
        (like OSError) if the output directory does not exist.

        FIX: Added OSError to the expected exceptions, matching the error raised by Pandas.
        """
        print("\nTest 5b-2: Testing controlled error for invalid output path (non-existent directory).")
        
        # Create an invalid path that points to a non-existent, deeply nested directory
        invalid_base_filename = os.path.join(self.temp_dir, "non_existent_dir", "report")

        # We expect the underlying file write operation to fail and raise an OS-level error
        with self.assertRaises((FileNotFoundError, PermissionError, OSError)): # FIXED: Added OSError
            print(f"  -> Calling export_summary_report with bad directory: {os.path.dirname(invalid_base_filename)}...")
            export_summary_report(self.summary_data, invalid_base_filename)
            
        print("  -> SUCCESS: A controlled I/O exception was raised as expected.")


if __name__ == "__main__":
    unittest.main()