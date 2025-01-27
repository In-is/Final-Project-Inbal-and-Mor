import os
import logging
from pathlib import Path
from typing import Optional

import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
VISUALIZATION_DIR = Path('./visualization/for_fun')
DATA_DIR = Path('./data')
RAW_DATA_DIR = DATA_DIR / 'raw'
COMPARISON_FILE = RAW_DATA_DIR / 'PLGG_vs_PHGG_Comparison.xlsx'

class ComparisonVisualizer:
    """Class for creating visualizations comparing PLGG and PHGG data."""
    
    def __init__(self):
        """Initialize the comparison visualizer."""
        self.data: Optional[pd.DataFrame] = None
        
        # Create output directory if it doesn't exist
        VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)
        
        # Set plot style
        plt.rcParams['figure.figsize'] = (10, 6)
    
    def load_data(self) -> None:
        """Load and validate the comparison data."""
        try:
            self.data = pd.read_excel(COMPARISON_FILE)
            logger.info("Comparison data loaded successfully")
        except Exception as e:
            logger.error(f"Error loading comparison data: {e}")
            raise

    def create_sunburst_chart(self) -> None:
        """Create and save sunburst chart visualization."""
        try:
            # Transform data for sunburst visualization
            sunburst_data = []
            for _, row in self.data.iterrows():
                sunburst_data.append([row['Category'], "PLGG", row['PLGG (Pediatric Low-Grade Glioma)']])
                sunburst_data.append([row['Category'], "PHGG", row['PHGG (Pediatric High-Grade Glioma)']])

            sunburst_df = pd.DataFrame(sunburst_data, columns=["Category", "Type", "Value"])

            fig = px.sunburst(sunburst_df, path=["Category", "Type", "Value"], 
                          title="PLGG vs PHGG Sunburst Visualization",
                          color_discrete_sequence=px.colors.qualitative.Set3)

            fig.update_layout(margin=dict(t=40, l=0, r=0, b=0))
            
            # Save the figure
            output_path = VISUALIZATION_DIR / "plgg_phgg_sunburst.html"
            fig.write_html(str(output_path))
            logger.info(f"Sunburst chart saved to {output_path}")
        except Exception as e:
            logger.error(f"Error creating sunburst chart: {e}")
            raise

    def create_word_cloud(self) -> None:
        """Create and save word cloud visualization."""
        try:
            # Combine all text into a single string
            text_data = " ".join(self.data['PLGG (Pediatric Low-Grade Glioma)'].astype(str).tolist() +
                             self.data['PHGG (Pediatric High-Grade Glioma)'].astype(str).tolist())

            # Generate word cloud
            wordcloud = WordCloud(width=800, height=400, 
                                background_color='black', 
                                colormap='cool').generate(text_data)

            # Plot the word cloud
            plt.figure(figsize=(10, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title("PLGG vs PHGG Word Cloud")
            
            # Save the figure
            output_path = VISUALIZATION_DIR / "plgg_phgg_wordcloud.png"
            plt.savefig(str(output_path), bbox_inches='tight', dpi=300)
            plt.close()
            logger.info(f"Word cloud saved to {output_path}")
        except Exception as e:
            logger.error(f"Error creating word cloud: {e}")
            raise

def main():
    """Main function to generate fun visualizations."""
    try:
        visualizer = ComparisonVisualizer()
        visualizer.load_data()
        visualizer.create_sunburst_chart()
        visualizer.create_word_cloud()
        logger.info("Fun visualizations completed successfully")
    except Exception as e:
        logger.error(f"Error in fun visualizations: {e}")
        raise

if __name__ == "__main__":
    main()