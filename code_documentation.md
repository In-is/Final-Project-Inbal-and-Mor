# Code Documentation: Glioma Analysis Pipeline

## Table of Contents
1. [Overview](#overview)
2. [Main Pipeline](#main-pipeline)
3. [Data Cleaning Module](#data-cleaning-module)
4. [HGG Analysis Module](#hgg-analysis-module)
5. [PLGG Analysis Module](#plgg-analysis-module)
6. [Joint PLGG-HGG Analysis Module](#joint-plgg-hgg-analysis-module)
7. [Predictive Modeling Module](#predictive-modeling-module)

## Overview
This project implements a comprehensive analysis pipeline for studying genetic mutations in High-Grade Gliomas (HGG) and Pediatric Low-Grade Gliomas (PLGG). The pipeline consists of multiple stages including data cleaning, individual analysis of HGG and PLGG data, joint analysis, and predictive modeling.

## Main Pipeline
**File: `main.py`**

The main pipeline orchestrates the entire analysis process in the following order:

1. **Data Cleaning**: Prepares and cleans the raw data
2. **HGG Analysis**: Analyzes High-Grade Glioma data
3. **PLGG Analysis**: Analyzes Pediatric Low-Grade Glioma data
4. **Joint Analysis**: Compares HGG and PLGG data
5. **Predictive Modeling**: Builds and evaluates prediction models
6. **Fun Visualizations**: Creates additional visualizations

### Key Functions:
- `run_pipeline()`: Main function that executes all analysis steps in sequence
- Each step is logged with start and completion messages
- Error handling ensures pipeline stability

## Data Cleaning Module
**File: `src/data_cleaning.py`**

This module prepares the raw data for analysis through several steps:

### Functions:
1. `setup_plot_params()`: 
   - Sets up matplotlib parameters for consistent visualization

2. `load_and_inspect_data(file_path)`:
   - Loads CSV file
   - Standardizes column names (lowercase, no spaces)
   - Displays basic dataset information
   - Returns cleaned DataFrame

3. `handle_missing_values(df, create_plots=True)`:
   - Identifies missing values
   - Creates visualizations:
     - Bar plot of missing value percentages
     - Heatmap of missing values
   - Fills missing values:
     - Numeric columns: median
     - Object columns: 'None'

4. `remove_duplicates(df)`:
   - Removes duplicate rows
   - Reports number of duplicates removed

5. `handle_outliers(df, create_plots=True)`:
   - Identifies outliers using IQR method
   - Creates box plots for numerical columns
   - Flags outliers with new columns
   - Does not remove outliers to preserve data

6. `convert_data_types(df)`:
   - Converts specific columns to categorical type
   - Currently handles: 'dipg_nbs_hgg', 'location', 'tumor_grade'

## HGG Analysis Module
**File: `src/data_analysis/data_analysis_hgg.py`**

Analyzes High-Grade Glioma genetic data using object-oriented approach:

### Classes:
1. `Gene`:
   - Represents individual genes
   - Properties: name, functional_class

2. `FunctionalClass`:
   - Groups genes by function
   - Methods:
     - `add_gene()`: Adds gene to class

3. `HGGAnalysis`:
   - Main analysis class
   - Key methods:
     - `load_data()`: Loads cleaned HGG data
     - `process_mutation_data()`: Calculates mutation totals
     - `create_mutation_heatmap()`: Visualizes mutation patterns
     - `analyze_top_genes()`: Identifies most mutated genes
     - `analyze_rare_mutations()`: Studies rare mutations (1-2 occurrences)

## PLGG Analysis Module
**File: `src/data_analysis/data_analysis_plgg.py`**

Analyzes Pediatric Low-Grade Glioma data:

### Class: PLGGAnalysis
Key methods:
1. `load_data()` & `load_split_data()`:
   - Loads main and supplementary PLGG data

2. `create_brain_location_heatmap()`:
   - Creates heatmap of gene frequencies by brain location
   - Shows mean frequency percentage

3. `analyze_tumor_types()`:
   - Analyzes gene frequencies by tumor type
   - Creates heatmap and identifies top genes
   - Saves results to text file

4. `analyze_pathway_involvement()`:
   - Creates stacked bar chart of gene frequencies
   - Shows pathway relationships

5. `analyze_therapy_availability()`:
   - Analyzes therapy options by pathway
   - Highlights RAS/MAPK and PI3K/AKT pathways

## Joint PLGG-HGG Analysis Module
**File: `src/data_analysis/data_analysis_joint_plgg_hgg.py`**

Compares PLGG and HGG data:

### Class: JointPLGGHGGAnalysis
Key methods:
1. `load_and_process_plgg_data()`:
   - Loads and processes PLGG data
   - Saves unique genes and mutations

2. `load_and_process_hgg_data()`:
   - Processes HGG data into comparable format
   - Logs mutation counts for key genes

3. `analyze_gene_overlap()`:
   - Finds shared genes between PLGG and HGG
   - Identifies unique genes in each type

4. `analyze_gene_frequencies()`:
   - Compares mutation frequencies
   - Creates comparative visualizations

5. `analyze_mutation_types()`:
   - Compares mutation patterns
   - Focuses on shared genes

## Predictive Modeling Module
**File: `src/data_analysis/predictmodels.py`**

Builds and evaluates machine learning models:

### Class: PredictiveModeling
Key methods:
1. `load_and_preprocess_data()`:
   - Loads cleaned HGG data
   - Encodes categorical variables

2. `prepare_train_test_data()`:
   - Splits data into training and test sets
   - Applies SMOTE for class balancing

3. `train_models()`:
   - Trains Random Forest and XGBoost models
   - Uses balanced data for training

4. Visualization methods:
   - `plot_feature_importance()`: Shows important features
   - `plot_confusion_matrix()`: Shows prediction accuracy
   - `plot_roc_curve()`: Shows model performance

5. `evaluate_models()`:
   - Generates comprehensive model comparison
   - Saves metrics and visualizations

### Model Details:
- **Random Forest**:
  - 100 trees
  - Used for baseline predictions

- **XGBoost**:
  - Advanced gradient boosting
  - Trained on SMOTE-balanced data

## Usage Instructions
1. Ensure all required data files are in the correct locations:
   - HGG data in `data/raw/HGG_DB.csv`
   - PLGG data in `data/raw/PLGG_DB.csv` and `PLGG_DB_2.csv`
   - Location data in `data/raw/PLGG_HGG_locations_with_mutations.csv`

2. Run the main pipeline:
   ```python
   python main.py
   ```

3. Results will be saved in:
   - `visualization/` directory (plots and figures)
   - `data/processed/` directory (cleaned data)
   - Individual analysis directories for specific results
