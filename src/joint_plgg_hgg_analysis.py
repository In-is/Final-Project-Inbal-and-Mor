import pandas as pd

plgg_file_path = "./data/raw/PLGG_DB.csv" 
plgg_data = pd.read_csv(plgg_file_path)
# Group by 'Gene' and 'Mutation Type', and keep unique rows
plgg_grouped = plgg_data.groupby(["Gene", "Mutation Type"], as_index=False).first()

# Display only the unique genes and mutations
print("PLGG: Unique Genes and Mutations:")
print(plgg_grouped[["Gene", "Mutation Type"]])

# Save to CSV
output_grouped_path = "PLGG_Unique_Genes_Mutations.csv"
plgg_grouped[["Gene", "Mutation Type"]].to_csv(output_grouped_path, index=False)
print(f"Unique genes and mutations saved to: {output_grouped_path}")


import pandas as pd

# Load the HGG dataset
hgg_file_path = "./data/processed/HGG_DB_cleaned.csv"
hgg_data = pd.read_csv(hgg_file_path)

# Display columns in the dataset
print("Columns in the dataset:")
print(hgg_data.columns)

# Automatically identify gene-related columns (from 5th column onward)
gene_columns = hgg_data.columns[5:]  # Adjust this range based on the structure of your dataset

# Melt the dataset for gene columns only
hgg_long = hgg_data.melt(
    id_vars=["sample", "location", "tumor_grade"],  # Columns to keep
    value_vars=gene_columns,  # Gene-related columns
    var_name="Gene",  # Name of the gene column
    value_name="Mutation Type"  # Name of the mutation column
)

# Convert gene and mutation names to lowercase
hgg_long["Gene"] = hgg_long["Gene"].str.lower()
hgg_long["Mutation Type"] = hgg_long["Mutation Type"].str.lower()

# Remove rows where Mutation Type is NaN
hgg_long = hgg_long[hgg_long["Mutation Type"].notna()]

# Drop duplicates to get unique Gene-Mutation combinations
hgg_unique = hgg_long[["Gene", "Mutation Type"]].drop_duplicates()

# Display the unique genes and mutations
print("Unique Genes and Mutations:")
print(hgg_unique)

# Save the processed data to a CSV file
output_path = "HGG_Unique_Genes_Mutations.csv"
hgg_unique.to_csv(output_path, index=False)
print(f"Processed data saved to: {output_path}")





import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Process PLGG dataset
plgg_file_path = "./data/raw/PLGG_DB_2.csv"
plgg_data = pd.read_csv(plgg_file_path)

# Normalize PLGG gene names
plgg_data["Gene"] = plgg_data["Gene"].str.lower().str.strip()
plgg_data["Gene"] = plgg_data["Gene"].str.replace(r"[^a-zA-Z0-9]", "", regex=True)

# Filter out non-gene entries if necessary
non_gene_entries = ["3_yrs", "sample", "location", "tumor_grade"]
plgg_data = plgg_data[~plgg_data["Gene"].isin(non_gene_entries)]

# Extract unique genes from PLGG
plgg_genes = set(plgg_data["Gene"].unique())

# Step 2: Process HGG dataset
hgg_file_path = "./data/processed/HGG_DB_cleaned.csv"
hgg_data = pd.read_csv(hgg_file_path)

# Automatically identify gene-related columns
gene_columns = [col for col in hgg_data.columns if col not in ["sample", "location", "tumor_grade", "3_yrs"]]

# Normalize HGG gene names
hgg_genes = set([col.lower().strip().replace(r"[^a-zA-Z0-9]", "") for col in gene_columns])

# Step 3: Compare genes
shared_genes = plgg_genes.intersection(hgg_genes)
unique_plgg_genes = plgg_genes.difference(hgg_genes)
unique_hgg_genes = hgg_genes.difference(plgg_genes)

# Display basic results
print(f"\nNumber of shared genes: {len(shared_genes)}")
print(f"Shared Genes: {sorted(shared_genes)}")
print(f"\nNumber of unique genes in PLGG: {len(unique_plgg_genes)}")
print(f"Number of unique genes in HGG: {len(unique_hgg_genes)}")

# Step 4: Calculate and compare frequencies
# PLGG Frequency
plgg_freq = (
    plgg_data[plgg_data["Gene"].isin(shared_genes)]
    .groupby("Gene")["Frequency (%)"]
    .mean()
    .reset_index()
    .rename(columns={"Frequency (%)": "PLGG Frequency"})
)

# HGG Frequency
hgg_long = hgg_data.melt(
    id_vars=["sample", "location", "tumor_grade"], 
    value_vars=gene_columns, 
    var_name="Gene", 
    value_name="Mutation Type"
)
hgg_long["Gene"] = hgg_long["Gene"].str.lower().str.strip()

# Calculate HGG Frequency
hgg_long["Frequency (%)"] = hgg_long["Mutation Type"].apply(lambda x: 1 if pd.notna(x) else 0)
hgg_freq = (
    hgg_long[hgg_long["Gene"].isin(shared_genes)]
    .groupby("Gene")["Frequency (%)"]
    .sum()  # Sum occurrences for each gene
    .reset_index()
    .rename(columns={"Frequency (%)": "HGG Frequency"})
)

# Merge PLGG and HGG frequencies
freq_comparison = pd.merge(plgg_freq, hgg_freq, on="Gene", how="inner")

# Display frequency comparison
print("\nGene Frequency Comparison (PLGG vs HGG):")
print(freq_comparison)

# Step 5: Plot frequency comparison
freq_comparison.plot(x="Gene", kind="bar", figsize=(12, 6), colormap="viridis", edgecolor="black")
plt.title("Gene Frequency Comparison: PLGG vs HGG", fontsize=16)
plt.xlabel("Gene", fontsize=12)
plt.ylabel("Frequency (%)", fontsize=12)
plt.xticks(rotation=45, fontsize=10)
plt.legend(title="Group", fontsize=10)
plt.tight_layout()
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

# Step 6: Compare Mutation Types for Shared Genes
# Filter PLGG and HGG for shared genes
plgg_shared_mutations = plgg_data[plgg_data["Gene"].isin(shared_genes)][["Gene", "Mutation Type"]]
hgg_shared_mutations = hgg_long[hgg_long["Gene"].isin(shared_genes)][["Gene", "Mutation Type"]]

# Normalize mutation types for comparison
plgg_shared_mutations["Mutation Type"] = plgg_shared_mutations["Mutation Type"].str.lower().str.strip()
hgg_shared_mutations["Mutation Type"] = hgg_shared_mutations["Mutation Type"].str.lower().str.strip()

# Group and count unique mutation types for each gene
plgg_mutation_counts = plgg_shared_mutations.groupby(["Gene", "Mutation Type"]).size().reset_index(name="PLGG Count")
hgg_mutation_counts = hgg_shared_mutations.groupby(["Gene", "Mutation Type"]).size().reset_index(name="HGG Count")

# Merge mutation counts to compare between PLGG and HGG
mutation_comparison = pd.merge(
    plgg_mutation_counts,
    hgg_mutation_counts,
    on=["Gene", "Mutation Type"],
    how="outer",
    suffixes=(" (PLGG)", " (HGG)")
).fillna(0)  # Replace NaN with 0 for easier comparison

# Display mutation comparison table
print("\nMutation Comparison for Shared Genes:")
print(mutation_comparison)

# Step 7: Plot Mutation Type Comparison for a Specific Gene (e.g., 'braf')
gene_to_plot = "braf"  # Specify the gene to visualize

gene_mutations = mutation_comparison[mutation_comparison["Gene"] == gene_to_plot]

if not gene_mutations.empty:
    gene_mutations.set_index("Mutation Type")[["PLGG Count", "HGG Count"]].plot(
        kind="bar", figsize=(12, 6), colormap="tab10", edgecolor="black"
    )
    plt.title(f"Mutation Type Comparison for {gene_to_plot.upper()}", fontsize=16)
    plt.xlabel("Mutation Type", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.legend(title="Group", fontsize=10)
    plt.tight_layout()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()
else:
    print(f"No mutation data available for {gene_to_plot}.")


import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import pandas as pd

# Load the PLGG data
plgg_file_path = "./data/raw/PLGG_DB.csv"  # Update path as needed
plgg_data = pd.read_csv(plgg_file_path)

# Filter shared genes with their brain locations
shared_genes = ["braf", "tp53", "nf1"]  # Define shared genes
plgg_data["Gene"] = plgg_data["Gene"].str.lower().str.strip()
plgg_filtered = plgg_data[plgg_data["Gene"].isin(shared_genes)]

# Define gene to brain location mapping from PLGG data
plgg_mapping = plgg_filtered.set_index("Gene")["Brain Location"].to_dict()

# Define example brain locations for HGG if needed
hgg_mapping = {
    "braf": "supratentorial",
    "tp53": "hemispheric",
    "nf1": "posterior fossa",
}

# Combine the mappings
combined_mapping = {gene: (plgg_mapping.get(gene, ""), hgg_mapping.get(gene, "")) for gene in shared_genes}

# Load the brain image
brain_image_path = "./data/raw/brain_image.png"
brain_img = plt.imread(brain_image_path)

# Plot the brain image
fig, ax = plt.subplots(figsize=(10, 8))
ax.imshow(brain_img, extent=[0, 10, 0, 10], aspect='auto')

# Define approximate positions for brain regions on the image
location_positions = {
    "midline": (5, 8),
    "supratentorial": (7, 6),
    "hemispheric": (5, 5),
    "posterior fossa": (3, 3),
    "optic pathway": (6, 4),
}

# Plot gene labels on the brain image
for gene, (plgg_loc, hgg_loc) in combined_mapping.items():
    if plgg_loc in location_positions:
        x, y = location_positions[plgg_loc]
        ax.text(x, y, f"{gene.upper()} (PLGG)", color="blue", fontsize=12, ha='center')
    if hgg_loc in location_positions:
        x, y = location_positions[hgg_loc]
        ax.text(x, y - 0.5, f"{gene.upper()} (HGG)", color="red", fontsize=12, ha='center')

# Finalize the plot
ax.axis('off')
plt.title("Shared Genes by Brain Location (PLGG vs HGG)", fontsize=16)
plt.show()



