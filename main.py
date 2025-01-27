import logging
from pathlib import Path
import sys
import os
import time

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import analysis modules
from src.data_analysis.data_cleaning import main as clean_data
from src.data_analysis.data_analysis_hgg import main as hgg_analysis
from src.data_analysis.data_analysis_plgg import main as plgg_analysis
from src.data_analysis.data_analysis_joint_plgg_hgg import JointPLGGHGGAnalysis
from src.data_analysis.predictmodels import PredictiveModeling
from src.for_fun.fun_images_generator import main as fun_visualizations

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
    5. Predictive Modeling
    6. Fun Visualizations
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

        # Step 5: Predictive Modeling
        logger.info("Starting predictive modeling analysis...")
        predictive_modeling = PredictiveModeling()
        predictive_modeling.main()
        logger.info("Predictive modeling analysis completed successfully")

        # Step 6: Fun Visualizations
        logger.info("Starting fun visualizations...")
        fun_visualizations()
        logger.info("Fun visualizations completed successfully")

        # Calculate and log total execution time
        execution_time = time.time() - start_time
        logger.info(f"Complete pipeline executed successfully in {execution_time:.2f} seconds")

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    run_pipeline()
