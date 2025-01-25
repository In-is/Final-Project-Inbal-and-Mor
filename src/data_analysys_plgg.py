import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Create products/plgganalysis directory if it doesn't exist
if not os.path.exists('./products/plgganalysis'):
    os.makedirs('./products/plgganalysis')

# Load the PLGG data
plgg_file_path = "./data/raw/PLGG_DB.csv" 
plgg_data = pd.read_csv(plgg_file_path)

# Prepare the data for the heatmap
# Group by Brain Location and Gene, and calculate the mean frequency
heatmap_data = plgg_data.groupby(["Brain Location", "Gene"])["Frequency (%)"].mean().unstack(fill_value=0)

# Create a heatmap
plt.figure(figsize=(14, 8))
sns.heatmap(
    heatmap_data,
    annot=True,  # Display values in the cells
    fmt=".1f",   # Format for displaying percentages
    cmap="YlGnBu",  # Colormap
    linewidths=0.5,
    cbar_kws={"label": "Mean Frequency (%)"}
)

# Customize the plot
plt.title("Heatmap of Gene Frequencies by Brain Location", fontsize=18)
plt.xlabel("Brain Location", fontsize=14)
plt.ylabel("Genes", fontsize=14)
plt.xticks(rotation=45, ha="right", fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig('./products/plgganalysis/brain_location_gene_heatmap.png')
plt.close()




import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the PLGG data
plgg_file_path = "./data/raw/PLGG_DB.csv"
plgg_data = pd.read_csv(plgg_file_path)

# Prepare the data for analysis
# Group by Tumor Types and Gene, and calculate the mean frequency
tumor_gene_data = plgg_data.groupby(["Tumor Types", "Gene"])["Frequency (%)"].mean().unstack(fill_value=0)

# Plot a heatmap to show the relationship
plt.figure(figsize=(14, 8))
sns.heatmap(
    tumor_gene_data,
    annot=True,  # Display the frequency values
    fmt=".1f",   # Format for percentages
    cmap="coolwarm",  # Colormap
    linewidths=0.5,
    cbar_kws={"label": "Mean Frequency (%)"}
)

# Customize the plot
plt.title("Gene Frequencies by Tumor Types", fontsize=18)
plt.xlabel("Tumor Types", fontsize=12)
plt.ylabel("Genes", fontsize=12)
plt.xticks(rotation=45, ha="right", fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig('./products/plgganalysis/tumor_types_gene_heatmap.png')
plt.close()

# Step 2: Find the top 5 genes for each tumor type
top_genes_per_tumor = (
    tumor_gene_data
    .apply(lambda x: x.nlargest(5))  # Select the top 5 genes for each tumor type
    .fillna(0)  # Replace NaN values with 0 for better display
    .astype(int)  # Convert to integer for readability
)

# Display the top genes per tumor type
print("Top 5 Genes for Each Tumor Type:")
print(top_genes_per_tumor)

# Save the top genes to a CSV file
top_genes_output_path = "./data/processed/PLGG_top_genes_by_tumor_type.csv"
top_genes_per_tumor.to_csv(top_genes_output_path)






import pandas as pd
import matplotlib.pyplot as plt

# Load the PLGG data
plgg_file_path = "./data/raw/PLGG_DB.csv" 
plgg_data = pd.read_csv(plgg_file_path)

# Aggregate the data by summing frequencies for each Pathway-Gene pair
pathway_gene_data = (
    plgg_data.groupby(["Pathway Involvement", "Gene"])["Frequency (%)"]
    .sum()  
    .reset_index()
)

# Pivot the data to create a structure suitable for a stacked bar chart
stacked_data = pathway_gene_data.pivot(
    index="Pathway Involvement", columns="Gene", values="Frequency (%)"
).fillna(0)

# Create a stacked bar chart
stacked_data.plot(
    kind="bar",
    stacked=True,
    figsize=(14, 8),
    colormap="viridis",
    edgecolor="black"
)

# Customize the plot
plt.title("Stacked Bar Chart of Gene Frequencies by Pathway Involvement", fontsize=18)
plt.xlabel("Pathway Involvement", fontsize=12)
plt.ylabel("Mean Frequency (%)", fontsize=12)
plt.xticks(rotation=45, ha="right", fontsize=10)
plt.legend(title="Genes", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)
plt.tight_layout()
plt.savefig('./products/plgganalysis/pathway_involvement_stacked.png')
plt.close()





import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the PLGG data
plgg_file_path = "./data/raw/PLGG_DB.csv" 
plgg_data = pd.read_csv(plgg_file_path)

# Step 1: Count the number of genes per pathway
pathway_counts = plgg_data.groupby("Pathway Involvement")["Gene"].nunique().reset_index(name="Gene Count")

# Step 2: Simulate therapy availability for pathways
# Update based on actual therapy data if available
pathway_counts["Therapy Available"] = pathway_counts["Pathway Involvement"].apply(
    lambda x: 1 if x.lower() in ["ras/mapk", "pi3k/akt"] else 0
)

# Step 3: Create a better bar plot with custom colors
plt.figure(figsize=(12, 6))
custom_palette = {1: "#9b59b6", 0: "#5dade2"} 
sns.barplot(
    data=pathway_counts,
    x="Pathway Involvement",
    y="Gene Count",
    hue="Therapy Available",
    palette=custom_palette
)

# Customize the plot
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
plt.tight_layout()

# Add annotations to the bars
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

plt.savefig('./products/plgganalysis/pathway_therapy_analysis.png')
plt.close()

# Step 4: Highlight pathways without therapy
no_therapy_pathways = pathway_counts[pathway_counts["Therapy Available"] == 0]
print("Pathways without available therapies:")
print(no_therapy_pathways)






import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the split table
file_path = "./data/raw/PLGG_DB_2.csv"
split_table = pd.read_csv(file_path)

# -------- Step 1: Define the chronological order for Age Group
age_order = ["0-5y", "6-10y", "11-15y", "14+", "16-20y", "Varied"]
split_table["Age Group"] = pd.Categorical(split_table["Age Group"], categories=age_order, ordered=True)

# -------- Graph 1: Stacked Bar Chart - Tumor Frequency by Age Group --------
tumor_by_age = split_table.groupby(["Age Group", "Tumor Type"]).size().reset_index(name="Count")
tumor_by_age_pivot = tumor_by_age.pivot(index="Age Group", columns="Tumor Type", values="Count").fillna(0)

tumor_by_age_pivot.plot(kind="bar", stacked=True, figsize=(12, 6), colormap="viridis", edgecolor="black")
plt.title("Tumor Frequency by Age Group (Stacked Bar Chart)", fontsize=16)
plt.xlabel("Age Group", fontsize=10)
plt.ylabel("Tumor Frequency", fontsize=10)
plt.xticks(rotation=45, fontsize=10)
plt.legend(title="Tumor Type", fontsize=10, bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig('./products/plgganalysis/age_tumor_frequency.png')
plt.close()

# -------- Graph 2: Line Chart - Average Mutation Frequency by Tumor Type Over Age Groups --------
avg_mutation_by_age_tumor = (
    split_table.groupby(["Age Group", "Tumor Type"])["Frequency (%)"].mean().reset_index()
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
plt.savefig('./products/plgganalysis/age_mutation_frequency.png')
plt.close()

