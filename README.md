# Brain Tumor Analysis Project

## Table of Contents
- [Project Overview](#project-overview)
  - [Analyze Genetic Data](#analyze-genetic-data)
  - [Predictive Modeling](#predictive-modeling)
  - [Data Visualization](#data-visualization)
  - [Clinical Insights](#clinical-insights)
- [Installation Guide](#installation-guide)
  - [Prerequisites](#prerequisites)
  - [Setting Up the Project](#setting-up-the-project)
- [Project Structure](#project-structure)
- [Configuration Files](#configuration-files)
- [Core Components](#core-components)
  - [Main Execution File](#main-execution-file)
  - [Source Code Modules](#source-code-modules)
  - [Testing](#testing)
- [Usage](#usage)

## Project Overview
This project is a comprehensive data analysis and machine learning pipeline focused on studying brain tumors, specifically comparing High-Grade Glioma (HGG) and Pediatric Low-Grade Glioma (PLGG). The project aims to:

1. **Analyze Genetic Data**: Compare and analyze genetic mutations between HGG and PLGG tumors to understand their differences and similarities.

2. **Predictive Modeling**: Develop machine learning models (Random Forest and XGBoost) to:
   - Predict tumor characteristics
   - Identify important genetic markers
   - Analyze tumor locations and their relationship with mutations

3. **Data Visualization**: Generate comprehensive visualizations to:
   - Compare mutation patterns between HGG and PLGG
   - Visualize tumor locations and their genetic profiles
   - Present model performance and predictions

4. **Clinical Insights**: Help researchers and clinicians better understand:
   - The genetic differences between HGG and PLGG
   - The relationship between tumor location and genetic mutations
   - Potential predictive markers for tumor classification

The project combines data cleaning, statistical analysis, machine learning, and visualization techniques to provide insights into brain tumor genetics and classification.

## Installation Guide

### Prerequisites
1. Install Python 3.12 or later:
   - Visit [Python's official website](https://www.python.org/downloads/)
   - Download and run the installer for your operating system
   - During installation, make sure to check "Add Python to PATH"

### Setting Up the Project
1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/In-is/Final-Project-Inbal-and-Mor.git
   cd Final-Project-Inbal-and-Mor
   ```

2. Create a virtual environment:
   ```bash
   # Navigate to the project directory
   python -m venv venv
   ```

3. Activate the virtual environment:
   ```bash
   # On Windows
   .\venv\Scripts\activate

   # On Mac/Linux
   source venv/bin/activate
   ```

4. Install project dependencies:
   ```bash
   pip install -e .
   ```

## Project Structure

```
├── data/                                    # Data directory
│   ├── raw/                                # Raw input data
│   └── processed/                          # Cleaned and processed data
├── logs/                                   # Log files directory
├── notebooks/                              # Jupyter notebooks for exploration
├── src/                                    # Source code
│   ├── data_analysis/                      # Analysis modules
│   │   ├── data_analysis_hgg.py           # HGG analysis
│   │   ├── data_analysis_plgg.py          # PLGG analysis
│   │   ├── data_analysis_joint_plgg_hgg.py # Joint PLGG-HGG analysis
│   │   └── predictmodels.py               # Predictive modeling
│   ├── data_cleaning.py                    # Data preprocessing and cleaning
│   └── for_fun/                           # Additional visualizations
├── tests/                                  # Test files
├── visualization/                          # Generated visualizations
└── main.py                                # Main execution script
```

## Configuration Files

- `pyproject.toml`: Main project configuration file that specifies:
  - Python version requirements (Python 3.12+)
  - Project dependencies
  - Development dependencies
  - Build system requirements
  - Code formatting settings

- `pytest.ini`: Configuration for pytest testing framework
- `setup.cfg`: Additional project configuration settings

## Core Components

### Main Execution File
`main.py`: Orchestrates the entire analysis pipeline, including:
- Data cleaning
- HGG Analysis
- PLGG Analysis
- Joint PLGG-HGG Analysis
- Predictive Modeling
- Visualization Generation

### Source Code Modules

#### Data Analysis (`src/data_analysis/`)
- `data_analysis_hgg.py`: Analyzes HGG
- `data_analysis_plgg.py`: Analyzes PLGG
- `data_analysis_joint_plgg_hgg.py`: Performs comparative analysis between HGG and PLGG
- `predictmodels.py`: Implements predictive modeling capabilities

#### Visualization (`src/for_fun/`)
- `fun_images_generator.py`: Creates additional visualizations and graphics

### Testing
The `tests/` directory contains unit tests and integration tests for various components of the project.

## Usage

To run the complete analysis pipeline:
```bash
python main.py
```

This will execute:
1. Data cleaning procedures
2. Individual analyses for HGG and PLGG
3. Joint comparative analysis
4. Predictive modeling
5. Generation of visualizations

Results and logs will be saved in their respective directories:
- Processed data: `data/processed/`
- Log files: `logs/`
- Visualizations: `visualization/`

For development and testing:
```bash
# Run tests
pytest

# Run linting
ruff check .
