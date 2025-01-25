import os
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
PRODUCTS_DIR = Path('./products/plgganalysis')
DATA_DIR = Path('./data')
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
RAW_DATA_DIR = DATA_DIR / 'raw'
PLGG_FILE = RAW_DATA_DIR / 'PLGG_DB.csv'
PLGG_FILE_2 = RAW_DATA_DIR / 'PLGG_DB_2.csv'

class PLGGAnalysis:
    """Class for analyzing Pediatric Low-Grade Glioma (PLGG) genetic data."""
    
    def __init__(self):
        """Initialize the PLGG analysis."""
        self.data: Optional[pd.DataFrame] = None
        self.split_data: Optional[pd.DataFrame] = None
        
        # Create output directory if it doesn't exist
        PRODUCTS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Set plot style
        plt.rcParams['figure.figsize'] = (12, 8)
    
    def load_data(self) -> None:
        """Load and validate the PLGG data."""
        try:
            self.data = pd.read_csv(PLGG_FILE)
            logger.info("PLGG data loaded successfully")
        except Exception as e:
            logger.error(f"Error loading PLGG data: {e}")
            raise

    def load_split_data(self) -> None:
        """Load and validate the split PLGG data."""
        try:
            self.split_data = pd.read_csv(PLGG_FILE_2)
            logger.info("Split PLGG data loaded successfully")
        except Exception as e:
            logger.error(f"Error loading split PLGG data: {e}")
            raise

    def create_brain_location_heatmap(self) -> None:
        """Create and save heatmap of gene frequencies by brain location."""
        try:
            heatmap_data = self.data.groupby(["Brain Location", "Gene"])["Frequency (%)"].mean().unstack(fill_value=0)
            
            plt.figure(figsize=(14, 8))
            sns.heatmap(
                heatmap_data,
                annot=True,
                fmt=".1f",
                cmap="YlGnBu",
                linewidths=0.5,
                cbar_kws={"label": "Mean Frequency (%)"}
            )
            
            plt.title("Heatmap of Gene Frequencies by Brain Location", fontsize=18)
            plt.xlabel("Brain Location", fontsize=14)
            plt.ylabel("Genes", fontsize=14)
            plt.xticks(rotation=45, ha="right", fontsize=10)
            plt.yticks(fontsize=10)
            plt.tight_layout()
            plt.savefig(PRODUCTS_DIR / 'brain_location_gene_heatmap.png')
            plt.close()
            logger.info("Brain location heatmap created successfully")
        except Exception as e:
            logger.error(f"Error creating brain location heatmap: {e}")
            raise

    def analyze_tumor_types(self) -> None:
        """Analyze and visualize gene frequencies by tumor types."""
        try:
            tumor_gene_data = self.data.groupby(["Tumor Types", "Gene"])["Frequency (%)"].mean().unstack(fill_value=0)
            
            # Create heatmap
            plt.figure(figsize=(14, 8))
            sns.heatmap(
                tumor_gene_data,
                annot=True,
                fmt=".1f",
                cmap="coolwarm",
                linewidths=0.5,
                cbar_kws={"label": "Mean Frequency (%)"}
            )
            
            plt.title("Gene Frequencies by Tumor Types", fontsize=18)
            plt.xlabel("Tumor Types", fontsize=12)
            plt.ylabel("Genes", fontsize=12)
            plt.xticks(rotation=45, ha="right", fontsize=10)
            plt.yticks(fontsize=10)
            plt.tight_layout()
            plt.savefig(PRODUCTS_DIR / 'tumor_types_gene_heatmap.png')
            plt.close()
            
            # Find and save top genes per tumor
            top_genes_per_tumor = (
                tumor_gene_data
                .apply(lambda x: x.nlargest(5))
                .fillna(0)
                .astype(int)
            )
            
            top_genes_per_tumor.to_csv(PRODUCTS_DIR / 'PLGG_top_genes_by_tumor_type.csv')
            logger.info("Tumor types analysis completed successfully")
        except Exception as e:
            logger.error(f"Error in tumor types analysis: {e}")
            raise

    def analyze_pathway_involvement(self) -> None:
        """Analyze and visualize pathway involvement data."""
        try:
            # Create stacked bar chart
            pathway_gene_data = (
                self.data.groupby(["Pathway Involvement", "Gene"])["Frequency (%)"]
                .sum()
                .reset_index()
            )
            
            stacked_data = pathway_gene_data.pivot(
                index="Pathway Involvement",
                columns="Gene",
                values="Frequency (%)"
            ).fillna(0)
            
            stacked_data.plot(
                kind="bar",
                stacked=True,
                figsize=(14, 8),
                colormap="viridis",
                edgecolor="black"
            )
            
            plt.title("Stacked Bar Chart of Gene Frequencies by Pathway Involvement", fontsize=18)
            plt.xlabel("Pathway Involvement", fontsize=12)
            plt.ylabel("Mean Frequency (%)", fontsize=12)
            plt.xticks(rotation=45, ha="right", fontsize=10)
            plt.legend(title="Genes", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)
            plt.tight_layout()
            plt.savefig(PRODUCTS_DIR / 'pathway_involvement_stacked.png')
            plt.close()
            logger.info("Pathway involvement analysis completed successfully")
        except Exception as e:
            logger.error(f"Error in pathway involvement analysis: {e}")
            raise

    def analyze_therapy_availability(self) -> None:
        """Analyze and visualize therapy availability by pathway."""
        try:
            pathway_counts = self.data.groupby("Pathway Involvement")["Gene"].nunique().reset_index(name="Gene Count")
            pathway_counts["Therapy Available"] = pathway_counts["Pathway Involvement"].apply(
                lambda x: 1 if x.lower() in ["ras/mapk", "pi3k/akt"] else 0
            )
            
            plt.figure(figsize=(12, 6))
            custom_palette = {1: "#9b59b6", 0: "#5dade2"}
            sns.barplot(
                data=pathway_counts,
                x="Pathway Involvement",
                y="Gene Count",
                hue="Therapy Available",
                palette=custom_palette
            )
            
            plt.title("Gene Counts and Therapy Availability by Pathway", fontsize=18, fontweight="bold")
            plt.xlabel("Pathway Involvement", fontsize=14, labelpad=10)
            plt.ylabel("Gene Count", fontsize=14, labelpad=10)
            plt.xticks(rotation=30, ha="right", fontsize=10)
            plt.yticks(fontsize=10)
            plt.grid(axis="y", linestyle="--", alpha=0.7)
            plt.legend(
                title="Therapy Availability",
                loc="upper right",
                labels=["No Therapy (Sky Blue)", "Therapy Available (Purple)"],
                fontsize=12
            )
            
            for i, row in pathway_counts.iterrows():
                x_pos = pathway_counts.index.get_loc(i)
                y_pos = row["Gene Count"]
                plt.text(
                    x=x_pos,
                    y=y_pos + 0.3,
                    s=int(y_pos),
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    color="black"
                )
            
            plt.tight_layout()
            plt.savefig(PRODUCTS_DIR / 'pathway_therapy_analysis.png')
            plt.close()
            
            # Save pathways without therapy information
            no_therapy_pathways = pathway_counts[pathway_counts["Therapy Available"] == 0]
            no_therapy_pathways.to_csv(PROCESSED_DATA_DIR / 'pathways_without_therapy.csv', index=False)
            logger.info("Therapy availability analysis completed successfully")
        except Exception as e:
            logger.error(f"Error in therapy availability analysis: {e}")
            raise

    def analyze_age_tumor_frequency(self) -> None:
        """Analyze and visualize tumor frequency by age group."""
        try:
            age_order = ["0-5y", "6-10y", "11-15y", "14+", "16-20y", "Varied"]
            self.split_data["Age Group"] = pd.Categorical(self.split_data["Age Group"], categories=age_order, ordered=True)
            
            tumor_by_age = self.split_data.groupby(["Age Group", "Tumor Type"]).size().reset_index(name="Count")
            tumor_by_age_pivot = tumor_by_age.pivot(index="Age Group", columns="Tumor Type", values="Count").fillna(0)
            
            tumor_by_age_pivot.plot(kind="bar", stacked=True, figsize=(12, 6), colormap="viridis", edgecolor="black")
            plt.title("Tumor Frequency by Age Group (Stacked Bar Chart)", fontsize=16)
            plt.xlabel("Age Group", fontsize=10)
            plt.ylabel("Tumor Frequency", fontsize=10)
            plt.xticks(rotation=45, fontsize=10)
            plt.legend(title="Tumor Type", fontsize=10, bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.grid(axis="y", linestyle="--", alpha=0.7)
            plt.tight_layout()
            plt.savefig(PRODUCTS_DIR / 'age_tumor_frequency.png')
            plt.close()
            logger.info("Age tumor frequency analysis completed successfully")
        except Exception as e:
            logger.error(f"Error in age tumor frequency analysis: {e}")
            raise

    def analyze_age_mutation_frequency(self) -> None:
        """Analyze and visualize mutation frequency by age group and tumor type."""
        try:
            avg_mutation_by_age_tumor = (
                self.split_data.groupby(["Age Group", "Tumor Type"])["Frequency (%)"].mean().reset_index()
            )
            
            plt.figure(figsize=(12, 6))
            sns.lineplot(
                data=avg_mutation_by_age_tumor,
                x="Age Group",
                y="Frequency (%)",
                hue="Tumor Type",
                marker="o",
                palette="tab10",
            )
            plt.title("Line Chart: Average Mutation Frequency by Tumor Type Over Age Groups", fontsize=16)
            plt.xlabel("Age Group", fontsize=10)
            plt.ylabel("Average Frequency (%)", fontsize=10)
            plt.xticks(rotation=45, fontsize=10)
            plt.legend(title="Tumor Type", fontsize=10, bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.grid(axis="y", linestyle="--", alpha=0.7)
            plt.tight_layout()
            plt.savefig(PRODUCTS_DIR / 'age_mutation_frequency.png')
            plt.close()
            logger.info("Age mutation frequency analysis completed successfully")
        except Exception as e:
            logger.error(f"Error in age mutation frequency analysis: {e}")
            raise

def main():
    """Main execution function."""
    try:
        analysis = PLGGAnalysis()
        analysis.load_data()
        analysis.load_split_data()
        analysis.create_brain_location_heatmap()
        analysis.analyze_tumor_types()
        analysis.analyze_pathway_involvement()
        analysis.analyze_therapy_availability()
        analysis.analyze_age_tumor_frequency()
        analysis.analyze_age_mutation_frequency()
        logger.info("PLGG analysis completed successfully")
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()
