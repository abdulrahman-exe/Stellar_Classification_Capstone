"""
Prediction Pipeline
Handles loading trained models and making predictions on new data
"""

import logging
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Tuple, List, Union
import sys
import traceback

# Add src to path for imports
# Consider using editable installs (pip install -e .) for better path management
sys.path.append(str(Path(__file__).parent.parent))

from config import (
    BEST_MODEL_FILE, LABEL_ENCODER_FILE, PREPROCESSOR_FILE,
    FEATURE_NAMES_FILE, METADATA_FILE, ALL_FEATURES, TARGET_COLUMN,
    LOG_FORMAT
)

# Setup logging
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

class PredictionPipeline:
    """
    Handles model loading and prediction on new data
    """

    def __init__(self):
        self.model = None           # The full scikit-learn/imblearn pipeline
        self.classifier = None      # The trained classifier model step (for SHAP)
        self.scaler_step = None     # The scaler step (for SHAP)
        self.label_encoder = None
        self.feature_names = None   # List of 13 feature names model expects
        self.metadata = None
        self.is_loaded = False

    def load_model(self) -> None:
        """
        Load the trained model pipeline and associated artifacts.
        Extracts scaler and classifier steps for potential use (e.g., SHAP).
        """
        logger.info("=" * 60)
        logger.info("LOADING TRAINED MODEL AND ARTIFACTS")
        logger.info("=" * 60)

        try:
            # --- Load Model Pipeline ---
            if not BEST_MODEL_FILE.exists():
                raise FileNotFoundError(f"Model file not found: {BEST_MODEL_FILE}")
            self.model = joblib.load(BEST_MODEL_FILE)
            logger.info(f"✅ Full Pipeline Model loaded from {BEST_MODEL_FILE}")

            # --- DETAILED DEBUGGING ---
            print("\n" + "="*20 + " DEBUGGING LOADED PIPELINE " + "="*20)
            print(f"Loaded object type: {type(self.model)}")
            if hasattr(self.model, 'steps'):
                print("Pipeline Steps (from .steps attribute):")
                for name, step_obj in self.model.steps:
                    print(f"  - Name: '{name}', Type: {type(step_obj)}")
            elif hasattr(self.model, 'named_steps'):
                 print("Pipeline Steps (from .named_steps attribute):")
                 print(f"  Keys: {list(self.model.named_steps.keys())}")
                 for name, step_obj in self.model.named_steps.items():
                    print(f"  - Name: '{name}', Type: {type(step_obj)}")
            else:
                print("Loaded object doesn't seem to be a Pipeline (no .steps or .named_steps).")
            print("="*60 + "\n")
            # --- END DEBUGGING ---

            # --- Extract Scaler and Classifier Steps ---
            if hasattr(self.model, 'named_steps'):
                # *** IMPORTANT: Check the DEBUG output and use the CORRECT name here ***
                self.scaler_step = self.model.named_steps.get('scaler') # Defaulting to 'scaler'
                self.classifier = self.model.named_steps.get('classifier')

                # Logging confirmations
                if self.classifier:
                    logger.info(f"✅ Extracted classifier step: {type(self.classifier).__name__}")
                else:
                    logger.warning("Classifier step ('classifier') not found in pipeline.")
                if self.scaler_step:
                    logger.info(f"✅ Extracted scaler step: {type(self.scaler_step).__name__}")
                else:
                    # This warning is likely appearing if the name isn't 'scaler'
                    logger.warning("Scaler step ('scaler') not found in pipeline.")
            else:
                logger.warning("Could not extract named steps from the loaded model object.")

            # --- Load Other Artifacts ---
            if not LABEL_ENCODER_FILE.exists():
                 raise FileNotFoundError(f"Label encoder file not found: {LABEL_ENCODER_FILE}")
            self.label_encoder = joblib.load(LABEL_ENCODER_FILE)
            logger.info(f"✅ Label encoder loaded from {LABEL_ENCODER_FILE}")

            if not FEATURE_NAMES_FILE.exists():
                 raise FileNotFoundError(f"Feature names file not found: {FEATURE_NAMES_FILE}")
            self.feature_names = joblib.load(FEATURE_NAMES_FILE)
            logger.info(f"✅ Feature names loaded from {FEATURE_NAMES_FILE}")
            if not isinstance(self.feature_names, list) or len(self.feature_names) != 13:
                 logger.warning(f"Loaded feature_names might be incorrect: {self.feature_names}")


            if not METADATA_FILE.exists():
                 raise FileNotFoundError(f"Metadata file not found: {METADATA_FILE}")
            self.metadata = joblib.load(METADATA_FILE)
            logger.info(f"✅ Metadata loaded from {METADATA_FILE}")

            self.is_loaded = True

            # --- Log Summary ---
            logger.info("\n📊 Model Information Summary:")
            logger.info(f"   Feature names expected (13): {self.feature_names}")
            logger.info(f"   Classes: {list(self.label_encoder.classes_)}")
            if 'class_mapping' in self.metadata:
                 logger.info(f"   Class mapping: {self.metadata['class_mapping']}")
            else:
                 logger.warning("   Class mapping not found in metadata.")

            logger.info("✅ Artifacts loaded successfully. Ready for predictions!")

        except FileNotFoundError as e:
            logger.error(f"❌ Artifact file not found: {str(e)}")
            logger.error("   Please ensure the training pipeline ran successfully and saved all artifacts.")
            raise
        except Exception as e:
            logger.error(f"❌ Error during artifact loading: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Engineers features from raw data (u, g, r, i, z, redshift).
        Must match the feature engineering in DataTransformation exactly.
        """
        logger.info("Engineering features for prediction...")
        df = data.copy() # Avoid modifying original DataFrame

        # Check if required raw columns exist
        required_raw = ['u', 'g', 'r', 'i', 'z', 'redshift']
        if not all(col in df.columns for col in required_raw):
            missing = set(required_raw) - set(df.columns)
            raise ValueError(f"Missing raw columns needed for feature engineering: {missing}")

        # 1. Create color indices
        df['u_g'] = df['u'] - df['g']
        df['g_r'] = df['g'] - df['r']
        df['r_i'] = df['r'] - df['i']
        df['i_z'] = df['i'] - df['z']

        # 2. Create aggregate features
        mag_cols = ['u', 'g', 'r', 'i', 'z']
        df['mag_mean'] = df[mag_cols].mean(axis=1)
        df['mag_std'] = df[mag_cols].std(axis=1)

        # 3. Create color variation
        color_cols = ['u_g', 'g_r', 'r_i', 'i_z']
        df['color_variation'] = df[color_cols].std(axis=1)

        # 4. Validate and reorder columns
        missing_engineered = set(self.feature_names) - set(df.columns)
        if missing_engineered:
            raise ValueError(f"Feature engineering failed, missing derived features: {missing_engineered}")

        extra_engineered = set(df.columns) - set(self.feature_names)
        if extra_engineered:
             logger.warning(f"Extra columns found after engineering: {extra_engineered}. Selecting only required features.")


        # Return DataFrame with columns in the exact order the model expects
        try:
             return df[self.feature_names]
        except KeyError as e:
             logger.error(f"Column mismatch after engineering. Expected: {self.feature_names}, Got: {list(df.columns)}")
             raise e


    def predict(self, data: Union[pd.DataFrame, np.ndarray, List]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on new data. Handles feature engineering if necessary.
        Input data can be raw (6 features) or already engineered (13 features).
        """
        if not self.is_loaded:
            raise ValueError("Model artifacts not loaded. Call load_model() first.")

        logger.info("Preprocessing and making predictions...")

        # --- Convert input to DataFrame ---
        if isinstance(data, (list, np.ndarray)):
            if isinstance(data, list):
                data = np.array(data)

            # Determine if input is raw (6) or engineered (13) based on columns
            if data.shape[1] == 6:
                logger.info("Input is array/list with 6 columns, assuming RAW features.")
                raw_feature_names = ['u', 'g', 'r', 'i', 'z', 'redshift']
                data_df = pd.DataFrame(data, columns=raw_feature_names)
            elif data.shape[1] == 13:
                logger.info("Input is array/list with 13 columns, assuming ENGINEERED features.")
                data_df = pd.DataFrame(data, columns=self.feature_names)
            else:
                raise ValueError(f"Input array/list has {data.shape[1]} columns. Expected 6 (raw) or 13 (engineered).")

        elif isinstance(data, pd.DataFrame):
            data_df = data.copy() # Work on a copy
        else:
            raise TypeError("Input data must be a pandas DataFrame, numpy array, or list.")

        # --- Engineer Features if Necessary ---
        # Check if all 13 required feature names are already columns
        if set(self.feature_names).issubset(data_df.columns):
            logger.info("Input DataFrame has 13 features. Assuming already engineered. Reordering columns.")
            # Ensure columns are in the correct order
            engineered_data = data_df[self.feature_names]
        # Check if it has the 6 raw features needed for engineering
        elif set(['u', 'g', 'r', 'i', 'z', 'redshift']).issubset(data_df.columns):
            logger.info("Input DataFrame has raw features. Starting feature engineering...")
            engineered_data = self._engineer_features(data_df)
        else:
             raise ValueError("Input DataFrame lacks required raw or engineered features.")


        # --- Make Predictions ---
        # self.model is the full pipeline (e.g., scaler -> smote -> classifier)
        # It expects the 13 UN-SCALED engineered features
        try:
            predictions = self.model.predict(engineered_data)
            probabilities = self.model.predict_proba(engineered_data)
        except Exception as e:
             logger.error(f"Error during model prediction: {e}")
             logger.error(f"Data shape passed to model: {engineered_data.shape}")
             logger.error(f"Data columns passed: {list(engineered_data.columns)}")
             raise e


        logger.info(f"✅ Predictions generated for {len(predictions)} samples.")

        return predictions, probabilities

    def predict_stellar_class(self, features: Union[List, np.ndarray]) -> Tuple[str, Dict[str, float]]:
        """
        Convenience function to predict a single instance.
        Input features can be 6 raw or 13 engineered.
        Returns predicted class name and probability dictionary.
        """
        if not self.is_loaded:
            raise ValueError("Model artifacts not loaded. Call load_model() first.")

        features_array = np.array(features).reshape(1, -1)

        # Use the main predict method
        pred_class_encoded, pred_proba = self.predict(features_array)

        # Extract results for the single instance
        pred_class_idx = pred_class_encoded[0]
        pred_proba_single = pred_proba[0]

        # Convert index to class name
        class_name = self.label_encoder.inverse_transform([pred_class_idx])[0]

        # Create probability dictionary
        probabilities = {self.label_encoder.classes_[i]: prob
                         for i, prob in enumerate(pred_proba_single)}

        return class_name, probabilities

    def predict_batch(self, data: Union[pd.DataFrame, np.ndarray, List]) -> pd.DataFrame:
        """
        Make predictions on a batch of data and return results as a DataFrame.
        """
        if not self.is_loaded:
            raise ValueError("Model artifacts not loaded. Call load_model() first.")

        logger.info(f"Making batch predictions for {len(data)} samples...")

        # Use the main predict method
        predictions, probabilities = self.predict(data)

        # Convert predictions to class names
        class_names = self.label_encoder.inverse_transform(predictions)

        # Create results DataFrame
        results = pd.DataFrame({
            'predicted_class': class_names,
            'predicted_class_encoded': predictions
        })

        # Add probability columns
        for i, class_name in enumerate(self.label_encoder.classes_):
            results[f'prob_{class_name}'] = probabilities[:, i]

        # Add confidence (max probability)
        results['confidence'] = np.max(probabilities, axis=1)

        logger.info("✅ Batch predictions completed successfully.")

        return results

    def get_model_info(self) -> Dict:
        """
        Return a dictionary containing information about the loaded model artifacts.
        """
        if not self.is_loaded:
            raise ValueError("Model artifacts not loaded. Call load_model() first.")

        return {
            'model_pipeline_object': self.model,
            'classifier_step_object': self.classifier,
            'scaler_step_object': self.scaler_step,
            'label_encoder_object': self.label_encoder,
            'feature_names_expected': self.feature_names,
            'classes_encoded': list(self.label_encoder.classes_),
            'class_mapping': self.metadata.get('class_mapping', 'Not found in metadata'),
            'n_features_expected': len(self.feature_names),
            'n_classes': len(self.label_encoder.classes_)
        }

# --- Main block for testing ---
def main():
    """
    Main function for testing the PredictionPipeline class.
    Loads artifacts and runs predictions on sample data.
    """
    logger.info("--- Testing PredictionPipeline ---")
    predictor = PredictionPipeline()

    try:
        predictor.load_model()

        # Test with 6 raw features (as the app would likely send)
        logger.info("\n--- Test Case 1: Predicting with 6 RAW features ---")
        raw_features = [20.0, 19.0, 18.0, 17.0, 16.0, 0.1] # Example data
        predicted_class, probabilities = predictor.predict_stellar_class(raw_features)
        print(f"Input (raw): {raw_features}")
        print(f"Predicted class: {predicted_class}")
        print(f"Probabilities: {probabilities}")

        # Test with 13 engineered features (e.g., from test data)
        logger.info("\n--- Test Case 2: Predicting with 13 ENGINEERED features ---")
        # Example data matching the order in self.feature_names
        engineered_features = [
            20.0, 19.0, 18.0, 17.0, 16.0, 0.1, # u, g, r, i, z, redshift
            1.0, 1.0, 1.0, 1.0,                # u_g, g_r, r_i, i_z
            18.0, np.std([20,19,18,17,16]), np.std([1,1,1,1]) # mag_mean, mag_std, color_variation
        ]
        if len(engineered_features) != 13:
             print("Error: Sample engineered features list length is not 13")
        else:
             predicted_class_eng, probabilities_eng = predictor.predict_stellar_class(engineered_features)
             print(f"Input (engineered): {engineered_features}")
             print(f"Predicted class: {predicted_class_eng}")
             print(f"Probabilities: {probabilities_eng}")

        # Test batch prediction with a small DataFrame
        logger.info("\n--- Test Case 3: Batch prediction with DataFrame (raw features) ---")
        batch_data_raw = pd.DataFrame([
            {'u': 21.0, 'g': 20.0, 'r': 19.5, 'i': 19.0, 'z': 18.8, 'redshift': 0.5},
            {'u': 18.0, 'g': 17.5, 'r': 17.0, 'i': 16.9, 'z': 16.8, 'redshift': 0.01},
        ])
        batch_results = predictor.predict_batch(batch_data_raw)
        print("Batch Prediction Results:")
        print(batch_results)


    except Exception as e:
        logger.error(f"Error during PredictionPipeline test: {e}")
        logger.error(traceback.format_exc())

    return predictor

if __name__ == "__main__":
    # When run directly, execute the main test function
    test_predictor = main()
    if test_predictor and test_predictor.is_loaded:
        logger.info("\nPredictionPipeline test completed.")
        # You could add more detailed checks here if needed
    else:
        logger.error("\nPredictionPipeline test FAILED.")