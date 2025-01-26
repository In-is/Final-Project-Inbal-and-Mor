# Ensure the project root is in the Python path
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import unittest
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Use absolute import
from src.data_analysis.data_analysis_plgg import (
    PLGGAnalysis, 
    VISUALIZATION_DIR, 
    PROCESSED_DATA_DIR, 
    RAW_DATA_DIR
)

class TestPLGGAnalysis(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        # Ensure visualization directory exists
        os.makedirs(VISUALIZATION_DIR, exist_ok=True)
        
        # Create an instance of PLGGAnalysis
        self.plgg_analysis = PLGGAnalysis()

    def test_initialization(self):
        """Test initialization of PLGGAnalysis class"""
        self.assertIsNone(self.plgg_analysis.data)
        self.assertIsNone(self.plgg_analysis.split_data)
        self.assertTrue(VISUALIZATION_DIR.exists())
        self.assertTrue(VISUALIZATION_DIR.is_dir())

    def test_visualization_directory_creation(self):
        """Test visualization directory creation"""
        self.assertTrue(VISUALIZATION_DIR.exists(), "Visualization directory should exist")
        self.assertTrue(VISUALIZATION_DIR.is_dir(), "Visualization path should be a directory")

    def test_data_loading(self):
        """Test data loading method"""
        # Check if the raw data file exists before attempting to load
        if os.path.exists(RAW_DATA_DIR / 'PLGG_DB.csv'):
            self.plgg_analysis.load_data()
            self.assertIsNotNone(self.plgg_analysis.data)
            self.assertFalse(self.plgg_analysis.data.empty)

    def test_split_data_loading(self):
        """Test split data loading method"""
        # Check if the split data file exists before attempting to load
        if os.path.exists(RAW_DATA_DIR / 'PLGG_DB_2.csv'):
            self.plgg_analysis.load_split_data()
            self.assertIsNotNone(self.plgg_analysis.split_data)
            self.assertFalse(self.plgg_analysis.split_data.empty)

    def test_visualization_methods(self):
        """Test visualization methods"""
        # Check if the raw data file exists before attempting to create visualizations
        if os.path.exists(RAW_DATA_DIR / 'PLGG_DB.csv'):
            self.plgg_analysis.load_data()

            # Test brain location heatmap
            self.plgg_analysis.create_brain_location_heatmap()
            brain_heatmap_file = VISUALIZATION_DIR / 'brain_location_gene_heatmap.png'
            self.assertTrue(brain_heatmap_file.exists(), "Brain location heatmap should be created")

            # Test tumor types analysis
            self.plgg_analysis.analyze_tumor_types()
            tumor_heatmap_file = VISUALIZATION_DIR / 'tumor_types_gene_heatmap.png'
            tumor_txt_file = VISUALIZATION_DIR / 'PLGG_top_genes_by_tumor_type.txt'
            self.assertTrue(tumor_heatmap_file.exists(), "Tumor types heatmap should be created")
            self.assertTrue(tumor_txt_file.exists(), "Tumor types text file should be created")

            # Test pathway involvement analysis
            self.plgg_analysis.analyze_pathway_involvement()
            pathway_plot_file = VISUALIZATION_DIR / 'pathway_involvement_stacked.png'
            self.assertTrue(pathway_plot_file.exists(), "Pathway involvement plot should be created")

if __name__ == '__main__':
    unittest.main()
