"""
Data Ingestion Component
Loads and validates raw data from star classification dataset
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Tuple
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config import RAW_DATA_FILE, RAW_FEATURES, INVALID_MAGNITUDE_VALUE, LOG_FORMAT

# Setup logging
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

class DataIngestion:
    """
    Handles data loading and initial validation
    """
    
    def __init__(self):
        self.raw_data_path = RAW_DATA_FILE
        self.data = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load raw data from CSV file
        Matches notebook: df = pd.read_csv("star_classification.csv")
        """
        try:
            logger.info(f"Loading data from {self.raw_data_path}")
            
            # Load the dataset
            self.data = pd.read_csv(self.raw_data_path)
            
            logger.info(f"Dataset loaded successfully!")
            logger.info(f"Shape: {self.data.shape}")
            
            return self.data
            
        except FileNotFoundError:
            logger.error(f"Data file not found: {self.raw_data_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def validate_data(self) -> dict:
        """
        Validate data quality and log metrics
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        logger.info("=" * 60)
        logger.info("DATA VALIDATION")
        logger.info("=" * 60)
        
        # Basic info
        logger.info(f"Dataset shape: {self.data.shape}")
        logger.info(f"Columns: {list(self.data.columns)}")
        
        # Check for missing values
        missing_values = self.data.isnull().sum()
        total_missing = missing_values.sum()
        
        logger.info("\nMissing Values:")
        for col, missing in missing_values.items():
            if missing > 0:
                logger.info(f"   {col}: {missing}")
        logger.info(f"Total missing values: {total_missing}")
        
        # Check for duplicates
        duplicates = self.data.duplicated().sum()
        logger.info(f"Duplicate rows: {duplicates}")
        
        # Check class distribution
        if 'class' in self.data.columns:
            class_dist = self.data['class'].value_counts()
            class_percentages = self.data['class'].value_counts(normalize=True) * 100
            
            logger.info("\nClass Distribution:")
            for cls, count in class_dist.items():
                logger.info(f"   {cls}: {count} ({class_percentages[cls]:.2f}%)")
        
        # Check for invalid magnitude data
        magnitude_cols = ['u', 'g', 'r', 'i', 'z']
        invalid_mask = (self.data[magnitude_cols] == INVALID_MAGNITUDE_VALUE).any(axis=1)
        invalid_count = invalid_mask.sum()
        
        logger.info(f"\nInvalid magnitude values (-9999): {invalid_count}")
        
        # Data types
        logger.info(f"\nData types:")
        for col, dtype in self.data.dtypes.items():
            logger.info(f"   {col}: {dtype}")
        
        # Statistical summary
        logger.info(f"\nStatistical Summary:")
        logger.info(self.data.describe().to_string())
        
        validation_results = {
            'shape': self.data.shape,
            'missing_values': total_missing,
            'duplicates': duplicates,
            'invalid_magnitudes': invalid_count,
            'class_distribution': class_dist.to_dict() if 'class' in self.data.columns else None
        }
        
        return validation_results
    
    def get_data_info(self) -> dict:
        """
        Get basic data information
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        return {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'dtypes': self.data.dtypes.to_dict(),
            'memory_usage': self.data.memory_usage(deep=True).sum()
        }
    
    def get_sample_data(self, n: int = 5) -> pd.DataFrame:
        """
        Get sample of the data
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        return self.data.head(n)

def main():
    """
    Main function for testing data ingestion
    """
    # Initialize data ingestion
    data_ingestion = DataIngestion()
    
    # Load data
    data = data_ingestion.load_data()
    
    # Validate data
    validation_results = data_ingestion.validate_data()
    
    # Get sample
    sample = data_ingestion.get_sample_data()
    print("\nSample data:")
    print(sample)
    
    return data, validation_results

if __name__ == "__main__":
    data, validation_results = main()