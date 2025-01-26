import logging
from pathlib import Path
import sys
import time

# Import analysis modules
from data_analysis.data_cleaning import main as clean_data
from data_analysis.data_analysis_hgg import main as hgg_analysis
from data_analysis.data_analysys_plgg import main as plgg_analysis
from data_analysis.data_analysis_joint_plgg_hgg import JointPLGGHGGAnalysis

# Configure paths
DATA_DIR = Path('./data')
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('analysis_pipeline.log')
    ]
)

logger = logging.getLogger(__name__)

def run_pipeline():
    """
    Execute the complete analysis pipeline in the following order:
    1. Data cleaning
    2. HGG Analysis
    3. PLGG Analysis
    4. Joint PLGG-HGG Analysis
    """
    start_time = time.time()
    
    try:
        # Step 1: Data Cleaning
        logger.info("Starting data cleaning process...")
        clean_data()
        logger.info("Data cleaning completed successfully")

        # Step 2: HGG Analysis
        logger.info("Starting HGG analysis...")
        hgg_analysis()
        logger.info("HGG analysis completed successfully")

        # Step 3: PLGG Analysis
        logger.info("Starting PLGG analysis...")
        plgg_analysis()
        logger.info("PLGG analysis completed successfully")

        # Step 4: Joint PLGG-HGG Analysis
        logger.info("Starting joint PLGG-HGG analysis...")
        joint_analysis = JointPLGGHGGAnalysis()
        joint_analysis.main()
        logger.info("Joint PLGG-HGG analysis completed successfully")

        # Calculate and log total execution time
        execution_time = time.time() - start_time
        logger.info(f"Complete pipeline executed successfully in {execution_time:.2f} seconds")

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    run_pipeline()
