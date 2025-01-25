import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os

# Create products/datacleaning directory if it doesn't exist
if not os.path.exists('./products/datacleaning'):
    os.makedirs('./products/datacleaning')


def setup_plot_params() -> None:
    """Set basic plot parameters"""
    plt.rcParams["figure.figsize"] = (10, 6)


def load_and_inspect_data(file_path):
    """Load and inspect the dataset"""
    # Read the CSV file
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


def handle_missing_values(df):
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

    # Visualize missing values
    plt.figure(figsize=(12, 6))
    sorted_summary = missing_value_summary.sort_values(by="Missing Percentage", ascending=False)
    sorted_summary["Missing Percentage"].plot(kind="bar", color="skyblue")
    plt.title("Percentage of Missing Values by Column", fontsize=16)
    plt.ylabel("Percentage", fontsize=12)
    plt.xlabel("Columns", fontsize=12)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('./products/datacleaning/missing_values_percentage.png')
    plt.close()

    # Heatmap of missing values
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.isnull(), yticklabels=False, cbar=True, cmap="coolwarm", cbar_kws={"label": "Missing Values"})
    plt.title("Missing Values Heatmap", fontsize=16)
    plt.savefig('./products/datacleaning/missing_values_heatmap.png')
    plt.close()

    # Replace missing values with 'None'
    df.fillna("None", inplace=True)

    print("\nMissing Values After Replacement:")
    print(df.isnull().sum())

    return df


def remove_duplicates(df):
    """Remove duplicate rows from the dataset"""
    duplicates = df.duplicated().sum()
    print("\nNumber of duplicate rows:", duplicates)

    if duplicates > 0:
        df = df.drop_duplicates()
        print("Duplicates removed. New shape:", df.shape)

    return df


def handle_outliers(df):
    """Handle outliers in numerical columns"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    if len(numeric_cols) == 0:
        print("\nNo numerical columns found in the dataset. Skipping outlier detection.")
        return df

    print("\nBasic Statistics:")
    print(df[numeric_cols].describe())

    plt.figure(figsize=(15, 5))
    df.boxplot(column=numeric_cols)
    plt.xticks(rotation=45)
    plt.title("Box Plots for Numerical Columns")
    plt.savefig('./products/datacleaning/numerical_boxplots.png')
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
    """Main function to run the data cleaning pipeline"""
    # Setup
    setup_plot_params()

    # Define input and output paths
    input_path = "./data/raw/HGG_DB.csv"
    output_path = "./data/processed/HGG_DB_cleaned.csv"

    # Execute cleaning pipeline
    df = load_and_inspect_data(input_path)
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = handle_outliers(df)
    df = convert_data_types(df)

    # Save cleaned data
    df.to_csv(output_path, index=False)
    print(f"\nCleaned dataset saved to '{output_path}'")


if __name__ == "__main__":
    main()
