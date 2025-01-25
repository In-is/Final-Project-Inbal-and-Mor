import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
PRODUCTS_DIR = Path('./products/hgganalysis')
DATA_DIR = Path('./data')
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
RAW_DATA_DIR = DATA_DIR / 'raw'

@dataclass
class Gene:
    """Class representing a gene with its functional classification."""
    name: str
    functional_class: Optional[str] = None

    def __repr__(self) -> str:
        return f"Gene(name={self.name}, functional_class={self.functional_class})"

@dataclass
class FunctionalClass:
    """Class representing a functional class containing multiple genes."""
    name: str
    genes: List[Gene]

    def add_gene(self, gene: Gene) -> None:
        """Add a gene to the functional class."""
        self.genes.append(gene)

    def __repr__(self) -> str:
        return f"FunctionalClass(name={self.name}, genes={[gene.name for gene in self.genes]})"

class HGGAnalysis:
    """Class for analyzing High-Grade Glioma (HGG) genetic data."""
    
    def __init__(self, db_path: Path, classes_path: Path):
        """Initialize the analysis with data paths.
        
        Args:
            db_path: Path to the cleaned HGG database
            classes_path: Path to the gene classes file
        """
        self.db_path = db_path
        self.classes_path = classes_path
        self.data: Optional[pd.DataFrame] = None
        self.classes: Optional[pd.DataFrame] = None
        self.functional_classes: Dict[str, FunctionalClass] = {}
        
        # Create output directory if it doesn't exist
        PRODUCTS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Set plot style
        plt.rcParams['figure.figsize'] = (10, 6)
        
    def load_data(self) -> None:
        """Load and validate the input data."""
        try:
            self.data = pd.read_csv(self.db_path)
            self.classes = pd.read_csv(self.classes_path)
            print(self.data.head())
            logger.info("Data loaded successfully")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def process_mutation_data(self) -> pd.DataFrame:
        """Process mutation data and calculate totals.
        
        Returns:
            DataFrame with processed mutation data
        """
        mutation_columns = self.data.columns[5:]
        self.data["Total Mutations"] = self.data[mutation_columns].notna().sum(axis=1)
        return self.data.groupby("location")[mutation_columns].apply(lambda x: x.notna().sum()).T

    def create_mutation_heatmap(self, mutation_by_location: pd.DataFrame) -> None:
        """Create and save mutation heatmap.
        
        Args:
            mutation_by_location: Processed mutation data by location
        """
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            mutation_by_location,
            annot=True,
            fmt="d",
            cmap="YlGnBu",
            linewidths=0.5,
            cbar_kws={"label": "Mutation Count"}
        )
        plt.title("Heatmap of Gene Mutations by Tumor Location", fontsize=16)
        plt.xlabel("Tumor Location", fontsize=12)
        plt.ylabel("Genes", fontsize=12)
        plt.xticks(rotation=45, ha="right", fontsize=12)
        plt.yticks(fontsize=10)
        plt.tight_layout()
        plt.savefig(PRODUCTS_DIR / 'mutation_heatmap.png')
        plt.close()

    def analyze_top_genes(self, mutation_by_location: pd.DataFrame) -> None:
        """Analyze and save top genes data.
        
        Args:
            mutation_by_location: Processed mutation data by location
        """
        top_genes_per_location = (
            mutation_by_location
            .apply(lambda x: x.nlargest(5))
            .fillna(0)
            .astype(int)
        )
        
        # Save top genes data
        top_genes_per_location.to_csv(PRODUCTS_DIR / 'HGG_top_genes_by_location.csv')
        
        # Create visualization for top 10 genes
        total_mutations = mutation_by_location.sum(axis=1).sort_values(ascending=False)
        top_10_genes = total_mutations.head(10)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x=top_10_genes.values, y=top_10_genes.index, palette="viridis")
        plt.title("Top 10 Genes with Most Mutations", fontsize=16)
        plt.xlabel("Number of Mutations", fontsize=12)
        plt.ylabel("Genes", fontsize=12)
        plt.tight_layout()
        plt.savefig(PRODUCTS_DIR / 'top_10_genes_barplot.png')
        plt.close()

    def process_functional_classes(self) -> None:
        """Process and analyze functional classes."""
        for _, row in self.classes.iterrows():
            fc_name = row["Functional Class"]
            genes = row.drop("Functional Class").dropna()
            
            if fc_name not in self.functional_classes:
                self.functional_classes[fc_name] = FunctionalClass(fc_name, [])
            
            for gene_name in genes:
                gene = Gene(name=gene_name, functional_class=fc_name)
                self.functional_classes[fc_name].add_gene(gene)

    def create_functional_class_visualizations(self) -> None:
        """Create visualizations for functional classes."""
        class_counts = {fc_name: len(fc.genes) for fc_name, fc in self.functional_classes.items()}
        
        plt.figure(figsize=(10, 6))
        sns.barplot(
            x=list(class_counts.values()),
            y=list(class_counts.keys()),
            palette="viridis"
        )
        plt.title("Number of Genes per Functional Class", fontsize=16)
        plt.xlabel("Number of Genes", fontsize=14)
        plt.ylabel("Functional Class", fontsize=14)
        plt.tight_layout()
        plt.savefig(PRODUCTS_DIR / 'functional_classes_barplot.png')
        plt.close()

    def analyze_rare_mutations(self) -> None:
         # Step 1: Load the dataset
        data_file_path = PROCESSED_DATA_DIR / "HGG_DB_cleaned.csv"
        data = pd.read_csv(data_file_path)

        # Step 2: Extract mutation columns
        mutation_columns = data.columns[5:]  # Assuming mutation columns start from index 5

        # Step 3: Count the occurrence of each mutation
        mutation_counts = data[mutation_columns].apply(pd.Series.value_counts).sum(axis=1).dropna().sort_values()

        # Step 4: Filter rare mutations (1-2 occurrences)
        rare_mutations = mutation_counts[mutation_counts <= 2]

        # Step 5: Create a table of rare mutations
        rare_mutations_table = rare_mutations.reset_index()
        rare_mutations_table.columns = ["Mutation", "Frequency"]

        # Display the table of rare mutations
        print("Rare Mutations (Frequency 1-2):")
        print(rare_mutations_table)

        # Step 6: Save the rare mutations table to a CSV file
        output_path = PROCESSED_DATA_DIR / "rare_mutations.csv"
        rare_mutations_table.to_csv(output_path, index=False)

        # Step 7: Create a scatter plot of mutation frequencies
        plt.figure(figsize=(12, 6))
        plt.scatter(
            mutation_counts.index,
            mutation_counts.values,
            alpha=0.7,
            c="blue",
            edgecolors="w",
            linewidth=0.5,
            s=50
        )
        plt.axhline(2, color="red", linestyle="--", label="Rare Mutation Threshold")
        plt.title("Mutation Frequencies", fontsize=14)
        plt.xlabel("Mutations", fontsize=6)
        plt.ylabel("Frequency", fontsize=14)
        plt.xticks(rotation=90, fontsize=6)
        plt.legend()
        plt.tight_layout()

        # Adjust layout and save
        plt.savefig(PRODUCTS_DIR / 'mutation_frequencies_scatter.png', 
                   bbox_inches='tight', 
                   dpi=300)
        plt.close()

    def create_mutation_by_location_class_heatmap(self) -> None:
        """Create heatmap showing mutation counts by location and functional class."""
        # Create gene to class mapping
        gene_to_class = {}
        for _, row in self.classes.iterrows():
            functional_class = row["Functional Class"]
            genes = row.drop("Functional Class").dropna()
            for gene in genes:
                gene_to_class[gene.strip()] = functional_class

        # Process data for heatmap
        gene_columns = self.data.columns[5:]
        db_table_melted = self.data.melt(
            id_vars=["location"],
            value_vars=gene_columns,
            var_name="Gene",
            value_name="Mutation Value"
        )

        # Map functional classes
        db_table_melted["Gene"] = db_table_melted["Gene"].str.strip()
        db_table_melted["Functional Class"] = db_table_melted["Gene"].map(gene_to_class)
        db_table_melted = db_table_melted.dropna(subset=["Functional Class"])

        # Count mutations
        db_table_melted["Mutation Value"] = db_table_melted["Mutation Value"].fillna("None")
        mutation_counts = (
            db_table_melted.groupby(["location", "Functional Class"])["Mutation Value"]
            .apply(lambda x: (x != "None").sum())
            .unstack(fill_value=0)
        )

        # Create heatmap
        plt.figure(figsize=(14, 8))
        sns.heatmap(
            mutation_counts,
            annot=True,
            fmt="d",
            cmap="YlGnBu",
            linewidths=0.5,
            cbar_kws={"label": "Number of Mutations"}
        )
        plt.title("Heatmap of Mutation Counts by Functional Class and Tumor Location", fontsize=16)
        plt.xlabel("Functional Class", fontsize=12)
        plt.ylabel("Tumor Location", fontsize=12)
        plt.xticks(rotation=45, ha="right", fontsize=10)
        plt.tight_layout()
        plt.savefig(PRODUCTS_DIR / 'mutation_counts_by_location_class_heatmap.png')
        plt.close()

    def create_age_location_mutation_plot(self) -> None:
        """Create plot showing average mutations by age and location."""
        mutation_stats = self.data.groupby(["3_yrs", "location"])["Total Mutations"].mean().reset_index()

        plt.figure(figsize=(14, 6))
        sns.barplot(
            data=mutation_stats,
            x="location",
            y="Total Mutations",
            hue="3_yrs",
            palette="viridis"
        )
        plt.title("Average Mutations by Age and Tumor Location", fontsize=16)
        plt.xlabel("Tumor Location", fontsize=14)
        plt.ylabel("Average Mutations", fontsize=14)
        plt.legend(title="3 Years Old")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(PRODUCTS_DIR / 'avg_mutations_by_age_location.png')
        plt.close()

    def analyze_functional_classes_by_grade(self) -> None:
        """Analyze and visualize functional classes distribution by tumor grade."""
        # Create gene to class mapping
        gene_to_class = {}
        for _, row in self.classes.iterrows():
            functional_class = row["Functional Class"]
            genes = row.drop("Functional Class").dropna()
            for gene in genes:
                gene_to_class[gene] = functional_class

        # Map functional classes to data
        functional_class_columns = self.data.columns[5:]
        self.data["Functional Class"] = self.data[functional_class_columns].apply(
            lambda row: [gene_to_class[gene] for gene in row.index 
                       if gene in gene_to_class and pd.notna(row[gene])],
            axis=1
        )

        # Process data
        data_exploded = self.data.explode("Functional Class")
        grade_class_counts = data_exploded.groupby(
            ["tumor_grade", "Functional Class"]
        ).size().unstack(fill_value=0)

        # Create normalized stacked bar plot
        grade_class_counts_normalized = grade_class_counts.div(
            grade_class_counts.sum(axis=0), axis=1
        )

        plt.figure(figsize=(12, 8))
        grade_class_counts_normalized.T.plot(
            kind="bar", 
            stacked=True, 
            figsize=(14, 8), 
            cmap="tab10"
        )
        plt.title("Distribution of Functional Classes by Tumor Grade", fontsize=16)
        plt.xlabel("Functional Class", fontsize=12)
        plt.ylabel("Proportion by Grade", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.legend(title="Tumor Grade", fontsize=12)
        plt.tight_layout()
        plt.savefig(PRODUCTS_DIR / 'functional_classes_by_grade.png')
        plt.close()

        # Create heatmap
        plt.figure(figsize=(14, 8))
        sns.heatmap(
            grade_class_counts,
            annot=True,
            fmt="d",
            cmap="YlGnBu",
            linewidths=0.5,
            cbar_kws={"label": "Number of Cases"}
        )
        plt.title("Heatmap of Functional Classes by Tumor Grade", fontsize=18)
        plt.xlabel("Functional Class", fontsize=12)
        plt.ylabel("Tumor Grade", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(PRODUCTS_DIR / 'functional_classes_grade_heatmap.png')
        plt.close()

def main():
    """Main execution function."""
    try:
        # Initialize analysisy
        analysis = HGGAnalysis(
            db_path=PROCESSED_DATA_DIR / 'HGG_DB_cleaned.csv',
            classes_path=RAW_DATA_DIR / 'Classes.organized.csv'
        )
        
        # Load data
        analysis.load_data()
        
        # Process mutations
        mutation_data = analysis.process_mutation_data()
        
        # Create visualizations
        analysis.create_mutation_heatmap(mutation_data)
        analysis.analyze_top_genes(mutation_data)
        analysis.process_functional_classes()
        analysis.create_functional_class_visualizations()
        analysis.create_mutation_by_location_class_heatmap()
        analysis.create_age_location_mutation_plot()
        analysis.analyze_functional_classes_by_grade()
        analysis.analyze_rare_mutations()
        
        logger.info("Analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        raise

if __name__ == "__main__":
    main() 
