"""
Data Transformation Component
Handles data cleaning, feature engineering, and train/test split
"""

import pandas as pd
import numpy as np
import logging
import joblib
from pathlib import Path
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config import (
    RAW_FEATURES, ENGINEERED_FEATURES, ALL_FEATURES, TARGET_COLUMN,
    INVALID_MAGNITUDE_VALUE, RANDOM_STATE, TEST_SIZE, VALIDATION_SIZE,
    TRAIN_DATA_FILE, VALIDATION_DATA_FILE, TEST_DATA_FILE, PREPROCESSOR_FILE, 
    LABEL_ENCODER_FILE, FEATURE_NAMES_FILE, METADATA_FILE,
    LOG_FORMAT
)

# Setup logging
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

class DataTransformation:
    """
    Handles data cleaning, feature engineering, and preprocessing
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.class_mapping = None
        
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean data by keeping only relevant features and removing invalid values
        """
        logger.info("=" * 60)
        logger.info("DATA CLEANING")
        logger.info("=" * 60)
        
        # Keep only relevant astrophysical features
        logger.info("Keeping only relevant astrophysical features...")
        df_clean = data[RAW_FEATURES].copy()
        
        # Remove invalid magnitude data (magnitude = -9999 indicates missing/invalid data)
        logger.info("Checking for invalid magnitude values...")
        magnitude_cols = ['u', 'g', 'r', 'i', 'z']
        invalid_mask = (df_clean[magnitude_cols] == INVALID_MAGNITUDE_VALUE).any(axis=1)
        invalid_count = invalid_mask.sum()
        
        logger.info(f"Found {invalid_count} rows with invalid data (-9999)")
        
        df_clean = df_clean[~invalid_mask].reset_index(drop=True)
        logger.info(f"✅ Clean dataset shape: {df_clean.shape}")
        logger.info(f"   Removed {invalid_count} invalid observations")
        
        return df_clean
    
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features
        Matches notebook feature engineering exactly
        """
        logger.info("=" * 60)
        logger.info("FEATURE ENGINEERING")
        logger.info("=" * 60)
        
        # Create a copy for feature engineering
        df_features = data.copy()
        
        # 1. Color Indices (spectral slopes between adjacent filters)
        logger.info("1️⃣ Creating color indices (u-g, g-r, r-i, i-z)...")
        df_features['u_g'] = df_features['u'] - df_features['g']
        df_features['g_r'] = df_features['g'] - df_features['r']
        df_features['r_i'] = df_features['r'] - df_features['i']
        df_features['i_z'] = df_features['i'] - df_features['z']
        logger.info("   ✅ Color indices created successfully")
        
        # 2. Aggregate magnitude statistics
        logger.info("2️⃣ Creating aggregate features...")
        df_features['mag_mean'] = df_features[['u', 'g', 'r', 'i', 'z']].mean(axis=1)
        df_features['mag_std'] = df_features[['u', 'g', 'r', 'i', 'z']].std(axis=1)
        logger.info("   ✅ Mean and std of magnitudes created")
        
        # 3. Color variation
        logger.info("3️⃣ Creating color variation metric...")
        df_features['color_variation'] = df_features[['u_g', 'g_r', 'r_i', 'i_z']].std(axis=1)
        logger.info("   ✅ Color variation created")
        
        logger.info(f"✅ Feature engineering complete!")
        logger.info(f"   Original features: {len(RAW_FEATURES) - 1}")
        logger.info(f"   Engineered features: {len(ENGINEERED_FEATURES)}")
        logger.info(f"   Total features: {df_features.shape[1] - 1} (excluding target)")
        
        return df_features
    
    def split_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Split data into train, validation, and test sets
        CRITICAL: Split BEFORE any scaling to prevent data leakage
        60% Train, 20% Validation, 20% Test
        """
        logger.info("=" * 60)
        logger.info("TRAIN/VALIDATION/TEST SPLIT - PREVENTING DATA LEAKAGE")
        logger.info("=" * 60)
        
        # Separate features and target
        X = data.drop(TARGET_COLUMN, axis=1)
        y = data[TARGET_COLUMN]
        
        # Encode target labels
        y_encoded = self.label_encoder.fit_transform(y)
        self.class_mapping = dict(zip(self.label_encoder.classes_, 
                                    self.label_encoder.transform(self.label_encoder.classes_)))
        
        logger.info(f"Class encoding: {self.class_mapping}")
        for i, cls in enumerate(self.label_encoder.classes_):
            logger.info(f"   {i} = {cls}")
        
        # First split: separate test set (20%)
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y_encoded, 
            test_size=TEST_SIZE, 
            random_state=RANDOM_STATE, 
            stratify=y_encoded
        )
        
        # Second split: separate train (60%) and validation (20%) from remaining 80%
        # Calculate validation size relative to the remaining data
        val_size_from_temp = VALIDATION_SIZE / (1 - TEST_SIZE)  # 0.2 / 0.8 = 0.25
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_from_temp,
            random_state=RANDOM_STATE,
            stratify=y_temp
        )
        
        logger.info(f"✅ Train set: {X_train.shape[0]} samples, {X_train.shape[1]} features ({X_train.shape[0]/len(X)*100:.1f}%)")
        logger.info(f"✅ Validation set: {X_val.shape[0]} samples, {X_val.shape[1]} features ({X_val.shape[0]/len(X)*100:.1f}%)")
        logger.info(f"✅ Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features ({X_test.shape[0]/len(X)*100:.1f}%)")
        
        # Log class distribution
        for split_name, X_split, y_split in [("Train", X_train, y_train), ("Validation", X_val, y_val), ("Test", X_test, y_test)]:
            logger.info(f"Class distribution in {split_name.lower()} set:")
            split_dist = pd.Series(y_split).value_counts().sort_index()
            for idx, count in split_dist.items():
                logger.info(f"   {self.label_encoder.classes_[idx]}: {count} ({count/len(y_split)*100:.2f}%)")
        
        logger.info("⚠️  IMPORTANT: Scaling will be done in the pipeline!")
        logger.info("   → Fit on train data, transform train, validation, and test")
        logger.info("   → This prevents data leakage from validation and test sets")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def fit_scaler(self, X_train: pd.DataFrame) -> None:
        """
        Fit scaler on training data only
        """
        logger.info("Fitting StandardScaler on training data...")
        self.scaler.fit(X_train)
        logger.info("✅ Scaler fitted successfully")
    
    def transform_data(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform data using fitted scaler
        """
        logger.info("Transforming data using fitted scaler...")
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        logger.info("✅ Data transformed successfully")
        
        return X_train_scaled, X_test_scaled
    
    def save_preprocessors(self) -> None:
        """
        Save preprocessor objects for later use
        """
        logger.info("Saving preprocessor objects...")
        
        # Save scaler
        joblib.dump(self.scaler, PREPROCESSOR_FILE)
        logger.info(f"✅ Scaler saved to {PREPROCESSOR_FILE}")
        
        # Save label encoder
        joblib.dump(self.label_encoder, LABEL_ENCODER_FILE)
        logger.info(f"✅ Label encoder saved to {LABEL_ENCODER_FILE}")
        
        # Save feature names
        joblib.dump(ALL_FEATURES, FEATURE_NAMES_FILE)
        logger.info(f"✅ Feature names saved to {FEATURE_NAMES_FILE}")
        
        # Save metadata
        metadata = {
            'class_mapping': self.class_mapping,
            'feature_names': ALL_FEATURES,
            'target_column': TARGET_COLUMN,
            'random_state': RANDOM_STATE,
            'test_size': TEST_SIZE
        }
        joblib.dump(metadata, METADATA_FILE)
        logger.info(f"✅ Metadata saved to {METADATA_FILE}")
    
    def save_processed_data(self, X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame, 
                          y_train: pd.Series, y_val: pd.Series, y_test: pd.Series) -> None:
        """
        Save processed train, validation, and test data
        """
        logger.info("Saving processed data...")
        
        # Add target back to features for saving
        train_data = X_train.copy()
        train_data[TARGET_COLUMN] = y_train
        
        val_data = X_val.copy()
        val_data[TARGET_COLUMN] = y_val
        
        test_data = X_test.copy()
        test_data[TARGET_COLUMN] = y_test
        
        # Save to CSV
        train_data.to_csv(TRAIN_DATA_FILE, index=False)
        val_data.to_csv(VALIDATION_DATA_FILE, index=False)
        test_data.to_csv(TEST_DATA_FILE, index=False)
        
        logger.info(f"✅ Train data saved to {TRAIN_DATA_FILE}")
        logger.info(f"✅ Validation data saved to {VALIDATION_DATA_FILE}")
        logger.info(f"✅ Test data saved to {TEST_DATA_FILE}")
    
    def load_preprocessors(self) -> None:
        """
        Load preprocessor objects
        """
        logger.info("Loading preprocessor objects...")
        
        self.scaler = joblib.load(PREPROCESSOR_FILE)
        self.label_encoder = joblib.load(LABEL_ENCODER_FILE)
        self.feature_names = joblib.load(FEATURE_NAMES_FILE)
        metadata = joblib.load(METADATA_FILE)
        self.class_mapping = metadata['class_mapping']
        
        logger.info("✅ Preprocessors loaded successfully")

def main():
    """
    Main function for testing data transformation
    """
    from data_ingestion import DataIngestion
    
    # Load data
    data_ingestion = DataIngestion()
    data = data_ingestion.load_data()
    
    # Transform data
    transformer = DataTransformation()
    
    # Clean data
    clean_data = transformer.clean_data(data)
    
    # Engineer features
    features_data = transformer.engineer_features(clean_data)
    
    # Split data
    X_train, X_test, y_train, y_test = transformer.split_data(features_data)
    
    # Fit scaler
    transformer.fit_scaler(X_train)
    
    # Save preprocessors
    transformer.save_preprocessors()
    
    # Save processed data
    transformer.save_processed_data(X_train, X_test, y_train, y_test)
    
    return transformer, X_train, X_test, y_train, y_test

if __name__ == "__main__":
    transformer, X_train, X_test, y_train, y_test = main()