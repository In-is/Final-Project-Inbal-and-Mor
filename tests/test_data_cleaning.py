import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add the src directory to the Python path so we can import the module
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data_cleaning import (
    handle_missing_values,
    remove_duplicates,
    handle_outliers,
    convert_data_types
)

class TestDataCleaning(unittest.TestCase):
    def setUp(self):
        """Set up test data"""
        # Create test directories if they don't exist
        os.makedirs('./visualization/datacleaning', exist_ok=True)
        os.makedirs('./data/raw', exist_ok=True)
        os.makedirs('./data/processed', exist_ok=True)
        
        # Create a sample DataFrame for testing with standardized column names
        self.test_data = pd.DataFrame({
            'dipg_nbs_hgg': ['DIPG', 'NBS', 'HGG', None, 'DIPG'],
            'location': ['Brain', 'Spine', None, 'Brain', 'Brain'],
            'tumor_grade': ['I', 'II', 'III', 'IV', 'I'],
            'age': [5, 10, 15, None, 5],
            'mutation_count': [1, 100, 3, 4, 1]  # 100 is an outlier
        })
        
        # Create a duplicate row
        self.data_with_duplicates = pd.concat([self.test_data, pd.DataFrame([self.test_data.iloc[0]])], ignore_index=True)

    def test_handle_missing_values(self):
        """Test missing values handling"""
        # Count initial missing values
        initial_missing = self.test_data.isnull().sum().sum()
        self.assertTrue(initial_missing > 0, "Test data should have missing values")

        # Process the data
        cleaned_df = handle_missing_values(self.test_data.copy(), create_plots=False)

        # Verify no missing values remain
        final_missing = cleaned_df.isnull().sum().sum()
        self.assertEqual(final_missing, 0, "All missing values should be replaced")

        # Verify missing values are replaced with 'None' for object columns
        none_count = cleaned_df['location'].astype(str).str.contains('None').sum()
        self.assertEqual(none_count, 1, "One 'None' value should be present in location column")

    def test_remove_duplicates(self):
        """Test duplicate removal"""
        # Initial row count
        initial_rows = len(self.data_with_duplicates)
        
        # Process the data
        cleaned_df = remove_duplicates(self.data_with_duplicates.copy())
        
        # Verify duplicates are removed
        self.assertTrue(len(cleaned_df) < initial_rows, "Duplicates should be removed")
        self.assertEqual(len(cleaned_df), len(cleaned_df.drop_duplicates()), "No duplicates should remain")

    def test_handle_outliers(self):
        """Test outlier detection and flagging"""
        # Process the data
        processed_df = handle_outliers(self.test_data.copy(), create_plots=False)
        
        # Check if outlier columns were created for numeric columns
        numeric_cols = self.test_data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            outlier_col = f"{col}_is_outlier"
            self.assertTrue(outlier_col in processed_df.columns, 
                          f"Outlier flag column {outlier_col} should be created")
        
        # Verify outlier detection in mutation_count
        self.assertTrue('mutation_count_is_outlier' in processed_df.columns)
        outlier_flags = processed_df['mutation_count_is_outlier'].sum()
        self.assertEqual(outlier_flags, 1, "One outlier should be detected in mutation_count")

    def test_convert_data_types(self):
        """Test data type conversion"""
        # Process the data
        converted_df = convert_data_types(self.test_data.copy())
        
        # Check if specified columns are converted to category
        categorical_cols = ['dipg_nbs_hgg', 'location', 'tumor_grade']
        existing_cols = [col for col in categorical_cols if col in converted_df.columns]
        
        for col in existing_cols:
            self.assertEqual(converted_df[col].dtype.name, 'category', 
                           f"{col} should be converted to category type")

    def test_complete_pipeline(self):
        """Test the entire data cleaning pipeline"""
        df = self.data_with_duplicates.copy()
        
        # Run through all cleaning steps
        df = handle_missing_values(df, create_plots=False)
        df = remove_duplicates(df)
        df = handle_outliers(df, create_plots=False)
        df = convert_data_types(df)
        
        # Verify final state
        self.assertEqual(df.isnull().sum().sum(), 0, "No missing values should remain")
        self.assertEqual(len(df), len(df.drop_duplicates()), "No duplicates should remain")
        self.assertTrue('mutation_count_is_outlier' in df.columns, "Outlier flags should be present")
        
        # Check categorical conversion for existing columns
        categorical_cols = ['dipg_nbs_hgg', 'location', 'tumor_grade']
        existing_cols = [col for col in categorical_cols if col in df.columns]
        for col in existing_cols:
            self.assertEqual(df[col].dtype.name, 'category', 
                           f"{col} should be converted to category type")

if __name__ == '__main__':
    unittest.main()
