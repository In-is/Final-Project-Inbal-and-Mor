import logging
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Set

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Constants
DATA_DIR = Path("./data")
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
VISUALIZATION_DIR = Path('./visualization/jointhgganalysis')

PLGG_FILE = RAW_DATA_DIR / "PLGG_DB.csv"
PLGG_FILE_2 = RAW_DATA_DIR / 'PLGG_DB_2.csv'
HGG_FILE = PROCESSED_DATA_DIR / "HGG_DB_cleaned.csv"
LOCATIONS_FILE = RAW_DATA_DIR / "PLGG_HGG_locations_with_mutations.csv"

class JointPLGGHGGAnalysis:
    """Class for analyzing and comparing Pediatric Low-Grade Glioma (PLGG) and High-Grade Glioma (HGG) genetic data."""
    
    def __init__(self):
        """Initialize the joint PLGG-HGG analysis."""
        self.plgg_data: Optional[pd.DataFrame] = None
        self.hgg_data: Optional[pd.DataFrame] = None
        self.plgg_grouped: Optional[pd.DataFrame] = None
        self.hgg_long: Optional[pd.DataFrame] = None
        self.shared_genes: Optional[Set[str]] = None
        self.locations_data: Optional[pd.DataFrame] = None
        
        # Create output directory if it doesn't exist
        VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)
        
    def load_and_process_plgg_data(self) -> None:
        """Load and process the PLGG dataset."""
        try:
            # Initial PLGG data loading
            self.plgg_data = pd.read_csv(PLGG_FILE)
            self.plgg_grouped = self.plgg_data.groupby(["Gene", "Mutation Type"], as_index=False).first()
            
            # Save unique genes and mutations
            output_grouped_path = VISUALIZATION_DIR / "PLGG_Unique_Genes_Mutations.csv"
            self.plgg_grouped[["Gene", "Mutation Type"]].to_csv(output_grouped_path, index=False)
            logger.info(f"PLGG unique genes and mutations saved to: {output_grouped_path}")
            
            # Process PLGG data for comparison
            plgg_data_2 = pd.read_csv(PLGG_FILE_2)
            plgg_data_2["Gene"] = plgg_data_2["Gene"].str.lower().str.strip()
            plgg_data_2["Gene"] = plgg_data_2["Gene"].str.replace(r"[^a-zA-Z0-9]", "", regex=True)
            
            # Filter out non-gene entries
            non_gene_entries = ["3_yrs", "sample", "location", "tumor_grade"]
            self.plgg_data = plgg_data_2[~plgg_data_2["Gene"].isin(non_gene_entries)]
            logger.info("PLGG data processed successfully")
        except Exception as e:
            logger.error(f"Error processing PLGG data: {e}")
            raise

    def load_and_process_hgg_data(self) -> None:
        """Load and process the HGG dataset."""
        try:
            self.hgg_data = pd.read_csv(HGG_FILE)
            logger.info("HGG data loaded successfully")
            
            # Display initial mutation counts for key genes
            for gene in ['braf', 'tp53', 'nf1']:
                if gene in self.hgg_data.columns:
                    mutations = self.hgg_data[gene].fillna('').astype(str)
                    mutations = mutations[mutations.str.strip() != '']
                    mutations = mutations[mutations.str.lower() != 'none']
                    if len(mutations) > 0:
                        logger.info(f"\nFound {len(mutations)} mutations for {gene.upper()} in HGG:")
                        for idx, mutation in mutations.items():
                            location = self.hgg_data.loc[idx, "location"]
                            logger.info(f"  - {mutation} at location {location}")
            
            # Identify gene columns
            gene_columns = [col for col in self.hgg_data.columns 
                          if col not in ["sample", "location", "tumor_grade", "3_yrs", "dipgnbshgg"]]
            
            # Create long format data
            self.hgg_long = self.hgg_data.melt(
                id_vars=["sample", "location", "tumor_grade"],
                value_vars=gene_columns,
                var_name="Gene",
                value_name="Mutation Type"
            )
            
            # Process gene names
            self.hgg_long["Gene"] = self.hgg_long["Gene"].str.lower()
            self.hgg_long["Mutation Type"] = self.hgg_long["Mutation Type"].fillna('').astype(str)
            
            # Remove rows with no mutations or 'None' mutations
            self.hgg_long = self.hgg_long[
                (self.hgg_long["Mutation Type"].str.strip() != '') & 
                (self.hgg_long["Mutation Type"].str.lower() != 'none')
            ]
            
            # Save unique genes and mutations
            hgg_unique = self.hgg_long[["Gene", "Mutation Type"]].drop_duplicates()
            output_path = VISUALIZATION_DIR / "HGG_Unique_Genes_Mutations.csv"
            hgg_unique.to_csv(output_path, index=False)
            logger.info(f"HGG unique genes and mutations saved to: {output_path}")
        except Exception as e:
            logger.error(f"Error processing HGG data: {e}")
            raise

    def analyze_gene_overlap(self) -> None:
        """Analyze the overlap between PLGG and HGG genes."""
        try:
            # Get unique genes from both datasets
            plgg_genes = set(self.plgg_data["Gene"].unique())
            gene_columns = [col for col in self.hgg_data.columns 
                          if col not in ["sample", "location", "tumor_grade", "3_yrs"]]
            hgg_genes = set([col.lower().strip().replace(r"[^a-zA-Z0-9]", "") for col in gene_columns])
            
            # Find overlapping and unique genes
            self.shared_genes = plgg_genes.intersection(hgg_genes)
            unique_plgg_genes = plgg_genes.difference(hgg_genes)
            unique_hgg_genes = hgg_genes.difference(plgg_genes)
            
            # Log results
            logger.info(f"Number of shared genes: {len(self.shared_genes)}")
            logger.info(f"Number of unique genes in PLGG: {len(unique_plgg_genes)}")
            logger.info(f"Number of unique genes in HGG: {len(unique_hgg_genes)}")
        except Exception as e:
            logger.error(f"Error analyzing gene overlap: {e}")
            raise

    def analyze_gene_frequencies(self) -> None:
        """Compare gene frequencies between PLGG and HGG."""
        try:
            # Calculate PLGG frequencies
            plgg_freq = (
                self.plgg_data[self.plgg_data["Gene"].isin(self.shared_genes)]
                .groupby("Gene")["Frequency (%)"]
                .mean()
                .reset_index()
                .rename(columns={"Frequency (%)": "PLGG Frequency"})
            )
            
            # Calculate HGG frequencies
            self.hgg_long["Frequency (%)"] = self.hgg_long["Mutation Type"].apply(
                lambda x: 1 if pd.notna(x) else 0
            )
            hgg_freq = (
                self.hgg_long[self.hgg_long["Gene"].isin(self.shared_genes)]
                .groupby("Gene")["Frequency (%)"]
                .sum()
                .reset_index()
                .rename(columns={"Frequency (%)": "HGG Frequency"})
            )
            
            # Merge frequencies
            freq_comparison = pd.merge(plgg_freq, hgg_freq, on="Gene", how="inner")
            
            # Create frequency comparison plot
            freq_comparison.plot(
                x="Gene",
                kind="bar",
                figsize=(12, 6),
                colormap="viridis",
                edgecolor="black"
            )
            plt.title("Gene Frequency Comparison: PLGG vs HGG", fontsize=16)
            plt.xlabel("Gene", fontsize=12)
            plt.ylabel("Frequency (%)", fontsize=12)
            plt.xticks(rotation=45, fontsize=10)
            plt.legend(title="Group", fontsize=10)
            plt.tight_layout()
            plt.grid(axis="y", linestyle="--", alpha=0.7)
            plt.savefig(VISUALIZATION_DIR / 'gene_frequency_comparison.png')
            plt.close()
            logger.info("Gene frequency comparison completed successfully")
        except Exception as e:
            logger.error(f"Error analyzing gene frequencies: {e}")
            raise

    def analyze_mutation_types(self, gene_to_plot: str = "braf") -> None:
        """Compare mutation types between PLGG and HGG for shared genes."""
        try:
            # Prepare mutation data
            plgg_shared_mutations = self.plgg_data[self.plgg_data["Gene"].isin(self.shared_genes)][["Gene", "Mutation Type"]]
            hgg_shared_mutations = self.hgg_long[self.hgg_long["Gene"].isin(self.shared_genes)][["Gene", "Mutation Type"]]
            
            # Normalize mutation types
            plgg_shared_mutations["Mutation Type"] = plgg_shared_mutations["Mutation Type"].str.lower().str.strip()
            hgg_shared_mutations["Mutation Type"] = hgg_shared_mutations["Mutation Type"].str.lower().str.strip()
            
            # Count mutations
            plgg_mutation_counts = plgg_shared_mutations.groupby(["Gene", "Mutation Type"]).size().reset_index(name="PLGG Count")
            hgg_mutation_counts = hgg_shared_mutations.groupby(["Gene", "Mutation Type"]).size().reset_index(name="HGG Count")
            
            # Merge mutation counts
            mutation_comparison = pd.merge(
                plgg_mutation_counts,
                hgg_mutation_counts,
                on=["Gene", "Mutation Type"],
                how="outer",
                suffixes=(" (PLGG)", " (HGG)")
            ).fillna(0)
            
            # Plot mutation comparison for specific gene
            gene_mutations = mutation_comparison[mutation_comparison["Gene"] == gene_to_plot]
            
            if not gene_mutations.empty:
                gene_mutations.set_index("Mutation Type")[["PLGG Count", "HGG Count"]].plot(
                    kind="bar",
                    figsize=(12, 6),
                    colormap="tab10",
                    edgecolor="black"
                )
                plt.title(f"Mutation Type Comparison for {gene_to_plot.upper()}", fontsize=16)
                plt.xlabel("Mutation Type", fontsize=12)
                plt.ylabel("Count", fontsize=12)
                plt.xticks(rotation=45, fontsize=10)
                plt.legend(title="Group", fontsize=10)
                plt.tight_layout()
                plt.grid(axis="y", linestyle="--", alpha=0.7)
                plt.savefig(VISUALIZATION_DIR / 'mutation_type_comparison.png')
                plt.close()
                logger.info(f"Mutation type comparison for {gene_to_plot} completed successfully")
            else:
                logger.warning(f"No mutation data available for {gene_to_plot}")
        except Exception as e:
            logger.error(f"Error analyzing mutation types: {e}")
            raise

    def analyze_brain_locations(self) -> None:
        """Analyze and visualize the prevalence of PLGG and HGG tumors by brain location."""
        try:
            # Load the locations data
            self.locations_data = pd.read_csv(LOCATIONS_FILE)
            logger.info("Brain locations data loaded successfully")

            # Group the data by tumor type
            plgg_data = self.locations_data[self.locations_data["Tumor_Type"] == "PLGG"]
            hgg_data = self.locations_data[self.locations_data["Tumor_Type"] == "HGG"]

            # Create the visualization
            plt.figure(figsize=(12, 6))
            
            # Plot PLGG and HGG data
            plt.bar(plgg_data["Location"], plgg_data["Prevalence_Percentage"], 
                   label="PLGG", alpha=0.7)
            plt.bar(hgg_data["Location"], hgg_data["Prevalence_Percentage"], 
                   label="HGG", alpha=0.7)

            # Customize the plot
            plt.title("Prevalence of PLGG and HGG Tumors by Brain Location")
            plt.xlabel("Brain Location")
            plt.ylabel("Prevalence Percentage")
            plt.xticks(rotation=45)
            plt.legend()
            plt.grid(axis="y", linestyle="--", alpha=0.5)
            plt.tight_layout()  # Ensure labels are visible

            # Save the plot
            output_path = VISUALIZATION_DIR / "brain_locations_prevalence.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Brain locations prevalence plot saved to: {output_path}")
        except Exception as e:
            logger.error(f"Error in brain locations analysis: {e}")
            raise

    def main(self) -> None:
        """Run all analyses."""
        try:
            self.load_and_process_plgg_data()
            self.load_and_process_hgg_data()
            self.analyze_gene_overlap()
            self.analyze_gene_frequencies()
            self.analyze_mutation_types()
            self.analyze_brain_locations()
            logger.info("Joint PLGG-HGG analysis completed successfully")
        except Exception as e:
            logger.error(f"Error in joint analysis: {e}")
            raise

if __name__ == "__main__":
    JointPLGGHGGAnalysis().main()
