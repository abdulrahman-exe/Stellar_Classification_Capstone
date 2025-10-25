"""
Configuration file for Stellar Classification ML Pipeline
Contains all hyperparameters, paths, and settings
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "saved_models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Data file paths
RAW_DATA_FILE = RAW_DATA_DIR / "star_classification.csv"
TRAIN_DATA_FILE = PROCESSED_DATA_DIR / "train.csv"
VALIDATION_DATA_FILE = PROCESSED_DATA_DIR / "validation.csv"
TEST_DATA_FILE = PROCESSED_DATA_DIR / "test.csv"
PREPROCESSOR_FILE = MODELS_DIR / "preprocessor.pkl"
LABEL_ENCODER_FILE = MODELS_DIR / "label_encoder.pkl"
BEST_MODEL_FILE = MODELS_DIR / "best_model.pkl"
FEATURE_NAMES_FILE = MODELS_DIR / "feature_names.pkl"
METADATA_FILE = MODELS_DIR / "metadata.pkl"

# Data configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2

# Features configuration
RAW_FEATURES = ['u', 'g', 'r', 'i', 'z', 'redshift', 'class']
ENGINEERED_FEATURES = ['u_g', 'g_r', 'r_i', 'i_z', 'mag_mean', 'mag_std', 'color_variation']
ALL_FEATURES = ['u', 'g', 'r', 'i', 'z', 'redshift'] + ENGINEERED_FEATURES
TARGET_COLUMN = 'class'

# Invalid data handling
INVALID_MAGNITUDE_VALUE = -9999

# Model hyperparameters (matching notebook exactly)
MODEL_CONFIGS = {
    'RandomForest': {
        'n_estimators': 100,
        'random_state': RANDOM_STATE,
        'n_jobs': -1
    },
    'XGBoost': {
        'random_state': RANDOM_STATE,
        'eval_metric': 'mlogloss',
        'use_label_encoder': False
    },
    'GradientBoosting': {
        'n_estimators': 100,
        'random_state': RANDOM_STATE
    },
    'SVM': {
        'probability': True,
        'random_state': RANDOM_STATE,
        'kernel': 'rbf'
    },
    'LogisticRegression': {
        'max_iter': 1000,
        'random_state': RANDOM_STATE
    }
}

# SMOTE configuration
SMOTE_CONFIG = {
    'random_state': RANDOM_STATE
}

# Visualization configuration
CUSTOM_PALETTE = ['#9400D3', '#00BFFF', '#FF6347']  # Purple (GALAXY), Blue (STAR), Red (QSO)
CLASS_COLORS = {'GALAXY': '#9400D3', 'STAR': '#00BFFF', 'QSO': '#FF6347'}

# Plotting configuration
PLOT_CONFIG = {
    'figure_dpi': 200,
    'seaborn_style': 'darkgrid',
    'axes_facecolor': '#faded9'
}

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# SHAP configuration
SHAP_CONFIG = {
    'sample_size': 1000,  # For efficiency
    'max_display': 15
}

# Model evaluation metrics
EVALUATION_METRICS = [
    'accuracy',
    'precision',
    'recall',
    'f1_score',
    'confusion_matrix'
]

# Create directories if they don't exist
def create_directories():
    """Create necessary directories if they don't exist"""
    directories = [
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        REPORTS_DIR / "figures",
        REPORTS_DIR / "figures" / "model_performance",
        REPORTS_DIR / "figures" / "validation"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

# Initialize directories
create_directories()
