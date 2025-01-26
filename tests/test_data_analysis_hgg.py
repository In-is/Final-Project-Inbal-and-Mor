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
from src.data_analysis.data_analysis_hgg import (
    HGGAnalysis, 
    Gene, 
    FunctionalClass, 
    PROCESSED_DATA_DIR, 
    VISUALIZATION_DIR
)

def find_data_file(filename):
    """Helper function to find data files."""
    possible_paths = [
        PROCESSED_DATA_DIR / filename,
        Path('./data/processed') / filename,
        Path(os.path.dirname(__file__)).parent / 'data' / 'processed' / filename
    ]
    for path in possible_paths:
        if path.exists():
            return path
    return None

class TestHGGAnalysis(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        # Ensure visualization directory exists
        os.makedirs(VISUALIZATION_DIR, exist_ok=True)

    def test_gene_class(self):
        """Test Gene class functionality."""
        gene = Gene("BRCA1", "DNA Repair")
        
        self.assertEqual(gene.name, "BRCA1")
        self.assertEqual(gene.functional_class, "DNA Repair")
        self.assertEqual(repr(gene), "Gene(name=BRCA1, functional_class=DNA Repair)")

    def test_functional_class(self):
        """Test FunctionalClass functionality."""
        gene1 = Gene("BRCA1")
        gene2 = Gene("BRCA2")
        
        fc = FunctionalClass("DNA Repair", [gene1])
        fc.add_gene(gene2)
        
        self.assertEqual(fc.name, "DNA Repair")
        self.assertEqual(len(fc.genes), 2)
        self.assertEqual(fc.genes[0].name, "BRCA1")
        self.assertEqual(fc.genes[1].name, "BRCA2")

    def test_visualization_directory_creation(self):
        """Test visualization directory creation."""
        self.assertTrue(VISUALIZATION_DIR.exists(), "Visualization directory should exist")
        self.assertTrue(VISUALIZATION_DIR.is_dir(), "Visualization path should be a directory")

if __name__ == '__main__':
    unittest.main()
