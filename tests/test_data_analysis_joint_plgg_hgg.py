# Ensure the project root is in the Python path
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import unittest
import pandas as pd
from pathlib import Path

# Use absolute import
from src.data_analysis.data_analysis_joint_plgg_hgg import (
    JointPLGGHGGAnalysis, 
    VISUALIZATION_DIR, 
    RAW_DATA_DIR, 
    PROCESSED_DATA_DIR
)

class TestJointPLGGHGGAnalysis(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        # Ensure visualization directory exists
        os.makedirs(VISUALIZATION_DIR, exist_ok=True)
        
        # Create an instance of JointPLGGHGGAnalysis
        self.joint_analysis = JointPLGGHGGAnalysis()

    def test_initialization(self):
        """Test initialization of JointPLGGHGGAnalysis class"""
        self.assertIsNone(self.joint_analysis.plgg_data)
        self.assertIsNone(self.joint_analysis.hgg_data)
        self.assertIsNone(self.joint_analysis.plgg_grouped)
        self.assertIsNone(self.joint_analysis.hgg_long)
        self.assertIsNone(self.joint_analysis.shared_genes)
        self.assertTrue(VISUALIZATION_DIR.exists())
        self.assertTrue(VISUALIZATION_DIR.is_dir())

    def test_visualization_directory_creation(self):
        """Test visualization directory creation"""
        self.assertTrue(VISUALIZATION_DIR.exists(), "Visualization directory should exist")
        self.assertTrue(VISUALIZATION_DIR.is_dir(), "Visualization path should be a directory")

    def test_data_loading_and_processing(self):
        """Test data loading and processing methods"""
        # Check if the required data files exist
        if os.path.exists(RAW_DATA_DIR / "PLGG_DB.csv") and \
           os.path.exists(RAW_DATA_DIR / "PLGG_DB_2.csv") and \
           os.path.exists(PROCESSED_DATA_DIR / "HGG_DB_cleaned.csv"):
            
            # Load and process PLGG data
            self.joint_analysis.load_and_process_plgg_data()
            self.assertIsNotNone(self.joint_analysis.plgg_data)
            self.assertIsNotNone(self.joint_analysis.plgg_grouped)
            
            # Check output files
            plgg_unique_path = VISUALIZATION_DIR / "PLGG_Unique_Genes_Mutations.csv"
            self.assertTrue(plgg_unique_path.exists(), "PLGG unique genes file should be created")
            
            # Load and process HGG data
            self.joint_analysis.load_and_process_hgg_data()
            self.assertIsNotNone(self.joint_analysis.hgg_data)
            self.assertIsNotNone(self.joint_analysis.hgg_long)
            
            # Check output files
            hgg_unique_path = VISUALIZATION_DIR / "HGG_Unique_Genes_Mutations.csv"
            self.assertTrue(hgg_unique_path.exists(), "HGG unique genes file should be created")

    def test_gene_overlap_analysis(self):
        """Test gene overlap analysis method"""
        # Load data first
        if os.path.exists(RAW_DATA_DIR / "PLGG_DB.csv") and \
           os.path.exists(RAW_DATA_DIR / "PLGG_DB_2.csv") and \
           os.path.exists(PROCESSED_DATA_DIR / "HGG_DB_cleaned.csv"):
            
            self.joint_analysis.load_and_process_plgg_data()
            self.joint_analysis.load_and_process_hgg_data()
            
            # Analyze gene overlap
            self.joint_analysis.analyze_gene_overlap()
            
            # Verify shared genes
            self.assertIsNotNone(self.joint_analysis.shared_genes)
            self.assertIsInstance(self.joint_analysis.shared_genes, set)
            self.assertTrue(len(self.joint_analysis.shared_genes) > 0, "There should be shared genes")

    def test_gene_mutation_calculation(self):
        """Test gene mutation calculations"""
        # Load data first
        if os.path.exists(RAW_DATA_DIR / "PLGG_DB.csv") and \
           os.path.exists(RAW_DATA_DIR / "PLGG_DB_2.csv") and \
           os.path.exists(PROCESSED_DATA_DIR / "HGG_DB_cleaned.csv"):
            
            self.joint_analysis.load_and_process_plgg_data()
            self.joint_analysis.load_and_process_hgg_data()
            self.joint_analysis.analyze_gene_overlap()
            
            # Verify gene mutation processing
            self.assertIsNotNone(self.joint_analysis.plgg_data)
            self.assertIsNotNone(self.joint_analysis.hgg_long)
            
            # Check data processing
            self.assertTrue(len(self.joint_analysis.plgg_data) > 0, "PLGG data should not be empty")
            self.assertTrue(len(self.joint_analysis.hgg_long) > 0, "HGG data should not be empty")
            
            # Verify gene columns are processed correctly
            self.assertTrue(all(isinstance(gene, str) for gene in self.joint_analysis.plgg_data["Gene"]), 
                            "Gene names should be strings")
            self.assertTrue(all(isinstance(gene, str) for gene in self.joint_analysis.hgg_long["Gene"]), 
                            "Gene names should be strings")
            
            # Verify mutation types are processed
            self.assertTrue(all(isinstance(mutation, str) for mutation in self.joint_analysis.plgg_data["Mutation Type"]), 
                            "Mutation types should be strings")
            self.assertTrue(all(isinstance(mutation, str) for mutation in self.joint_analysis.hgg_long["Mutation Type"]), 
                            "Mutation types should be strings")

    def test_gene_overlap_calculation(self):
        """Test gene overlap calculation"""
        # Load data first
        if os.path.exists(RAW_DATA_DIR / "PLGG_DB.csv") and \
           os.path.exists(RAW_DATA_DIR / "PLGG_DB_2.csv") and \
           os.path.exists(PROCESSED_DATA_DIR / "HGG_DB_cleaned.csv"):
            
            self.joint_analysis.load_and_process_plgg_data()
            self.joint_analysis.load_and_process_hgg_data()
            
            # Analyze gene overlap
            self.joint_analysis.analyze_gene_overlap()
            
            # Verify shared genes calculation
            self.assertIsNotNone(self.joint_analysis.shared_genes)
            self.assertIsInstance(self.joint_analysis.shared_genes, set)
            
            # Check shared genes conditions
            plgg_genes = set(self.joint_analysis.plgg_data["Gene"].unique())
            gene_columns = [col for col in self.joint_analysis.hgg_data.columns 
                          if col not in ["sample", "location", "tumor_grade", "3_yrs"]]
            hgg_genes = set([col.lower().strip() for col in gene_columns])
            
            # Verify overlap calculation
            calculated_shared_genes = plgg_genes.intersection(hgg_genes)
            self.assertEqual(
                self.joint_analysis.shared_genes, 
                calculated_shared_genes, 
                "Shared genes calculation should match"
            )

    def test_gene_frequencies_calculation(self):
        """Test gene frequencies calculation"""
        # Load data first
        if os.path.exists(RAW_DATA_DIR / "PLGG_DB.csv") and \
           os.path.exists(RAW_DATA_DIR / "PLGG_DB_2.csv") and \
           os.path.exists(PROCESSED_DATA_DIR / "HGG_DB_cleaned.csv"):
            
            self.joint_analysis.load_and_process_plgg_data()
            self.joint_analysis.load_and_process_hgg_data()
            self.joint_analysis.analyze_gene_overlap()
            
            # Prepare for frequency calculation
            plgg_freq = (
                self.joint_analysis.plgg_data[
                    self.joint_analysis.plgg_data["Gene"].isin(self.joint_analysis.shared_genes)
                ]
                .groupby("Gene")["Frequency (%)"]
                .mean()
                .reset_index()
            )
            
            # Calculate HGG frequencies
            self.joint_analysis.hgg_long["Frequency (%)"] = self.joint_analysis.hgg_long["Mutation Type"].apply(
                lambda x: 1 if pd.notna(x) else 0
            )
            hgg_freq = (
                self.joint_analysis.hgg_long[
                    self.joint_analysis.hgg_long["Gene"].isin(self.joint_analysis.shared_genes)
                ]
                .groupby("Gene")["Frequency (%)"]
                .sum()
                .reset_index()
            )
            
            # Verify frequency calculations
            self.assertTrue(len(plgg_freq) > 0, "PLGG frequency calculation should not be empty")
            self.assertTrue(len(hgg_freq) > 0, "HGG frequency calculation should not be empty")
            
            # Check frequency range
            self.assertTrue(
                all(0 <= freq <= 100 for freq in plgg_freq["Frequency (%)"] if pd.notna(freq)), 
                "PLGG frequencies should be between 0 and 100"
            )
            self.assertTrue(
                all(0 <= freq <= len(self.joint_analysis.hgg_data) for freq in hgg_freq["Frequency (%)"]), 
                "HGG frequencies should be between 0 and total sample count"
            )

if __name__ == '__main__':
    unittest.main()
