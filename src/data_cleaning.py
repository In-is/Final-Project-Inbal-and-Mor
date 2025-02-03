import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants for directories
DATA_DIR = Path('./data')
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
VISUALIZATION_DIR = Path('./visualization/datacleaning')

# Create required directories
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)


def setup_plot_params() -> None:
    """Set basic plot parameters"""
    plt.rcParams["figure.figsize"] = (10, 6)


def load_and_inspect_data(file_path: Path) -> pd.DataFrame:
    """Load and inspect the dataset
    
    Args:
        file_path: Path to the input CSV file
        
    Returns:
        pd.DataFrame: Loaded and initially processed dataframe
    """
    # Read the CSV file
    logger.info(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)

    # Display basic information about the dataset
    print("\nDataset Info:")
    df.info()

    # Standardize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("[^a-z0-9_]", "", regex=True)

    print("Cleaned column names:", df.columns.tolist())
    print(f"\nDataset Shape: {df.shape[0]} rows and {df.shape[1]} columns")
    print("\nFirst 5 Rows:")
    print(df.head().to_string(index=False))
    print("\nColumn Names:")
    print(", ".join(df.columns))

    num_duplicates = df.duplicated().sum()
    print(f"\nNumber of Duplicate Rows: {num_duplicates}")

    return df


def handle_missing_values(df, create_plots=True):
    """Handle missing values in the dataset"""
    # Identify missing values
    missing_value_summary = pd.DataFrame(
        {
            "Non-Missing Count": df.notnull().sum(),
            "Missing Count": df.isnull().sum(),
            "Missing Percentage": (df.isnull().sum() / len(df)) * 100,
        }
    )

    print("\nMissing Value Summary:")
    print(missing_value_summary[missing_value_summary["Missing Count"] > 0])

    if create_plots:
        # Visualize missing values
        plt.figure(figsize=(12, 6))
        sorted_summary = missing_value_summary.sort_values(by="Missing Percentage", ascending=False)
        sorted_summary["Missing Percentage"].plot(kind="bar", color="skyblue")
        plt.title("Percentage of Missing Values by Column", fontsize=16)
        plt.ylabel("Percentage", fontsize=12)
        plt.xlabel("Columns", fontsize=12)
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(VISUALIZATION_DIR / 'missing_values_percentage.png')
        plt.close()

        # Heatmap of missing values
        plt.figure(figsize=(12, 8))
        sns.heatmap(df.isnull(), yticklabels=False, cbar=True, cmap="coolwarm", cbar_kws={"label": "Missing Values"})
        plt.title("Missing Values Heatmap", fontsize=16)
        plt.savefig(VISUALIZATION_DIR / 'missing_values_heatmap.png')
        plt.close()

    # Replace missing values with 'None' for object columns and median for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    object_cols = df.select_dtypes(exclude=[np.number]).columns
    
    for col in numeric_cols:
        df[col].fillna(df[col].median(), inplace=True)
    for col in object_cols:
        df[col].fillna("None", inplace=True)

    print("\nMissing Values After Replacement:")
    print(df.isnull().sum())

    return df


def remove_duplicates(df):
    """Remove duplicate rows from the dataset"""
    initial_rows = len(df)
    df = df.drop_duplicates()
    removed_rows = initial_rows - len(df)

    print(f"\nRemoved {removed_rows} duplicate rows.")
    print(f"Dataset shape after removing duplicates: {df.shape}")

    return df


def handle_outliers(df, create_plots=True):
    """Handle outliers in numerical columns"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    if len(numeric_cols) == 0:
        print("\nNo numerical columns found in the dataset. Skipping outlier detection.")
        return df

    print("\nBasic Statistics:")
    print(df[numeric_cols].describe())

    if create_plots:
        plt.figure(figsize=(15, 5))
        df[numeric_cols].boxplot()
        plt.xticks(rotation=45)
        plt.title("Box Plots for Numerical Columns")
        plt.savefig(VISUALIZATION_DIR / 'numerical_boxplots.png')
        plt.close()

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        print(f"\nOutliers in column {col}:")
        print(outliers[[col]])

        df[f"{col}_is_outlier"] = ((df[col] < lower_bound) | (df[col] > upper_bound)).astype(int)

    print("\nColumns with outlier flags added:")
    print([col for col in df.columns if "_is_outlier" in col])

    return df


def convert_data_types(df):
    """Convert specific columns to categorical type"""
    print("\nColumn Names:")
    print(df.columns.tolist())

    categorical_cols = [col for col in ["dipg_nbs_hgg", "location", "tumor_grade"] if col in df.columns]

    if categorical_cols:
        df[categorical_cols] = df[categorical_cols].astype("category")
        print(f"\nConverted columns to 'category': {categorical_cols}")
    else:
        print("\nNo relevant columns found for conversion to 'category'.")

    return df


def main():
    """Main function to run the data cleaning pipeline."""
    # Setup plot parameters
    setup_plot_params()
    
    # Load and inspect data
    input_file = RAW_DATA_DIR / 'HGG_DB.csv'
    df = load_and_inspect_data(input_file)

    # Handle missing values
    df = handle_missing_values(df, create_plots=True)

    # Remove duplicates
    df = remove_duplicates(df)

    # Handle outliers
    df = handle_outliers(df, create_plots=True)

    # Convert data types
    df = convert_data_types(df)

    # Save cleaned data
    output_file = PROCESSED_DATA_DIR / 'HGG_DB_cleaned.csv'
    logger.info(f"Saving cleaned data to {output_file}")
    df.to_csv(output_file, index=False)
    logger.info("Data cleaning completed successfully")


if __name__ == "__main__":
    main()
