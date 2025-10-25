"""
Data Quality Check and Validation
Validates data quality and provides detailed statistics
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from config import (
    RAW_DATA_FILE, RAW_FEATURES, INVALID_MAGNITUDE_VALUE, 
    CUSTOM_PALETTE, CLASS_COLORS, LOG_FORMAT
)

# Setup logging
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

class DataValidator:
    """
    Validates data quality and provides detailed statistics
    """
    
    def __init__(self, data_path: str = None):
        self.data_path = data_path or RAW_DATA_FILE
        self.data = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load data for validation
        """
        logger.info(f"Loading data from {self.data_path}")
        self.data = pd.read_csv(self.data_path)
        logger.info(f"Data loaded successfully! Shape: {self.data.shape}")
        return self.data
    
    def check_basic_info(self) -> Dict[str, Any]:
        """
        Check basic data information
        Matches notebook basic info checks exactly
        """
        logger.info("=" * 60)
        logger.info("BASIC DATA INFORMATION")
        logger.info("=" * 60)
        
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        info = {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'dtypes': self.data.dtypes.to_dict(),
            'memory_usage': self.data.memory_usage(deep=True).sum()
        }
        
        logger.info(f"Dataset shape: {info['shape']}")
        logger.info(f"Columns: {info['columns']}")
        logger.info(f"Memory usage: {info['memory_usage'] / 1024**2:.2f} MB")
        
        return info
    
    def check_missing_values(self) -> Dict[str, Any]:
        """
        Check for missing values
        Matches notebook missing value checks exactly
        """
        logger.info("=" * 60)
        logger.info("MISSING VALUES CHECK")
        logger.info("=" * 60)
        
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        missing_values = self.data.isnull().sum()
        total_missing = missing_values.sum()
        
        logger.info("Missing Values:")
        for col, missing in missing_values.items():
            if missing > 0:
                logger.info(f"   {col}: {missing}")
        logger.info(f"Total missing values: {total_missing}")
        
        # Check for duplicates
        duplicates = self.data.duplicated().sum()
        logger.info(f"Duplicate rows: {duplicates}")
        
        return {
            'missing_by_column': missing_values.to_dict(),
            'total_missing': total_missing,
            'duplicates': duplicates
        }
    
    def check_class_distribution(self) -> Dict[str, Any]:
        """
        Check class distribution
        Matches notebook class distribution checks exactly
        """
        logger.info("=" * 60)
        logger.info("CLASS DISTRIBUTION CHECK")
        logger.info("=" * 60)
        
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        if 'class' not in self.data.columns:
            logger.warning("No 'class' column found in data")
            return {}
        
        class_dist = self.data['class'].value_counts()
        class_percentages = self.data['class'].value_counts(normalize=True) * 100
        
        logger.info("Class Distribution:")
        for cls, count in class_dist.items():
            logger.info(f"   {cls}: {count} ({class_percentages[cls]:.2f}%)")
        
        return {
            'class_counts': class_dist.to_dict(),
            'class_percentages': class_percentages.to_dict()
        }
    
    def check_invalid_magnitudes(self) -> Dict[str, Any]:
        """
        Check for invalid magnitude values
        Matches notebook invalid magnitude checks exactly
        """
        logger.info("=" * 60)
        logger.info("INVALID MAGNITUDE VALUES CHECK")
        logger.info("=" * 60)
        
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        magnitude_cols = ['u', 'g', 'r', 'i', 'z']
        invalid_by_column = {}
        total_invalid = 0
        
        for col in magnitude_cols:
            if col in self.data.columns:
                invalid_count = (self.data[col] == INVALID_MAGNITUDE_VALUE).sum()
                invalid_by_column[col] = invalid_count
                total_invalid += invalid_count
                logger.info(f"   {col}: {invalid_count} invalid values")
        
        # Check for any invalid values across all magnitude columns
        invalid_mask = (self.data[magnitude_cols] == INVALID_MAGNITUDE_VALUE).any(axis=1)
        rows_with_invalid = invalid_mask.sum()
        
        logger.info(f"Total rows with invalid magnitude values: {rows_with_invalid}")
        
        return {
            'invalid_by_column': invalid_by_column,
            'total_invalid': total_invalid,
            'rows_with_invalid': rows_with_invalid
        }
    
    def check_data_types(self) -> Dict[str, Any]:
        """
        Check data types and ranges
        Matches notebook data type checks exactly
        """
        logger.info("=" * 60)
        logger.info("DATA TYPES AND RANGES CHECK")
        logger.info("=" * 60)
        
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        dtypes_info = {}
        ranges_info = {}
        
        for col in self.data.columns:
            dtype = self.data[col].dtype
            dtypes_info[col] = str(dtype)
            
            if pd.api.types.is_numeric_dtype(self.data[col]):
                min_val = self.data[col].min()
                max_val = self.data[col].max()
                ranges_info[col] = {'min': min_val, 'max': max_val}
                logger.info(f"   {col}: {dtype} (range: {min_val:.4f} to {max_val:.4f})")
            else:
                unique_vals = self.data[col].nunique()
                ranges_info[col] = {'unique_values': unique_vals}
                logger.info(f"   {col}: {dtype} ({unique_vals} unique values)")
        
        return {
            'dtypes': dtypes_info,
            'ranges': ranges_info
        }
    
    def check_statistics_by_class(self) -> Dict[str, Any]:
        """
        Check statistics by stellar class
        Matches notebook class statistics exactly
        """
        logger.info("=" * 60)
        logger.info("STATISTICS BY STELLAR CLASS")
        logger.info("=" * 60)
        
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        if 'class' not in self.data.columns:
            logger.warning("No 'class' column found in data")
            return {}
        
        numeric_cols = ['u', 'g', 'r', 'i', 'z', 'redshift']
        available_cols = [col for col in numeric_cols if col in self.data.columns]
        
        class_stats = {}
        
        for cls in self.data['class'].unique():
            logger.info(f"\n{'='*60}")
            logger.info(f"{cls}:")
            logger.info(f"{'='*60}")
            
            class_data = self.data[self.data['class'] == cls][available_cols]
            stats = class_data.describe()
            
            logger.info(stats.to_string())
            class_stats[cls] = stats.to_dict()
        
        return class_stats
    
    def check_data_quality_score(self) -> float:
        """
        Calculate overall data quality score
        """
        logger.info("=" * 60)
        logger.info("DATA QUALITY SCORE")
        logger.info("=" * 60)
        
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        score = 100.0
        
        # Check missing values (penalty: 1 point per 0.1% missing)
        missing_info = self.check_missing_values()
        missing_percentage = (missing_info['total_missing'] / (self.data.shape[0] * self.data.shape[1])) * 100
        score -= missing_percentage * 10
        
        # Check duplicates (penalty: 1 point per 0.1% duplicates)
        duplicate_percentage = (missing_info['duplicates'] / self.data.shape[0]) * 100
        score -= duplicate_percentage * 10
        
        # Check invalid magnitudes (penalty: 1 point per 0.1% invalid)
        invalid_info = self.check_invalid_magnitudes()
        invalid_percentage = (invalid_info['rows_with_invalid'] / self.data.shape[0]) * 100
        score -= invalid_percentage * 10
        
        # Ensure score is not negative
        score = max(0, score)
        
        logger.info(f"Data Quality Score: {score:.2f}/100")
        
        if score >= 90:
            logger.info("✅ Excellent data quality")
        elif score >= 80:
            logger.info("✅ Good data quality")
        elif score >= 70:
            logger.info("⚠️  Fair data quality")
        else:
            logger.info("❌ Poor data quality - needs attention")
        
        return score
    
    def generate_quality_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive data quality report
        """
        logger.info("=" * 80)
        logger.info("GENERATING DATA QUALITY REPORT")
        logger.info("=" * 80)
        
        if self.data is None:
            self.load_data()
        
        report = {
            'basic_info': self.check_basic_info(),
            'missing_values': self.check_missing_values(),
            'class_distribution': self.check_class_distribution(),
            'invalid_magnitudes': self.check_invalid_magnitudes(),
            'data_types': self.check_data_types(),
            'class_statistics': self.check_statistics_by_class(),
            'quality_score': self.check_data_quality_score()
        }
        
        logger.info("✅ Data quality report generated successfully")
        
        return report
    
    def save_quality_report(self, report: Dict[str, Any], output_file: str = "data_quality_report.json") -> None:
        """
        Save quality report to file
        """
        import json
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Convert the report
        json_report = json.loads(json.dumps(report, default=convert_numpy))
        
        with open(output_file, 'w') as f:
            json.dump(json_report, f, indent=2)
        
        logger.info(f"✅ Quality report saved to {output_file}")

def main():
    """
    Main function for testing data validation
    """
    # Create validator
    validator = DataValidator()
    
    # Load data
    data = validator.load_data()
    
    # Generate quality report
    report = validator.generate_quality_report()
    
    # Save report
    validator.save_quality_report(report)
    
    return validator, report

if __name__ == "__main__":
    validator, report = main()
