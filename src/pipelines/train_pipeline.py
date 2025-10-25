"""
Training Pipeline
Orchestrates the complete training workflow from data loading to model saving
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from components.data_ingestion import DataIngestion
from components.data_transformation import DataTransformation
from components.model_trainer import ModelTrainer
from config import LOG_FORMAT

# Setup logging
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

class TrainingPipeline:
    """
    Complete training pipeline orchestrating all components
    """
    
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()
        
    def run_pipeline(self) -> dict:
        """
        Run the complete training pipeline
        """
        logger.info("=" * 80)
        logger.info("STELLAR CLASSIFICATION TRAINING PIPELINE")
        logger.info("=" * 80)
        
        try:
            # Step 1: Data Ingestion
            logger.info("\n" + "="*60)
            logger.info("STEP 1: DATA INGESTION")
            logger.info("="*60)
            
            data = self.data_ingestion.load_data()
            validation_results = self.data_ingestion.validate_data()
            
            # Step 2: Data Transformation
            logger.info("\n" + "="*60)
            logger.info("STEP 2: DATA TRANSFORMATION")
            logger.info("="*60)
            
            # Clean data
            clean_data = self.data_transformation.clean_data(data)
            
            # Engineer features
            features_data = self.data_transformation.engineer_features(clean_data)
            
            # Split data (CRITICAL: Before any scaling)
            X_train, X_val, X_test, y_train, y_val, y_test = self.data_transformation.split_data(features_data)
            
            # Fit scaler on training data only
            self.data_transformation.fit_scaler(X_train)
            
            # Save preprocessors
            self.data_transformation.save_preprocessors()
            
            # Save processed data
            self.data_transformation.save_processed_data(X_train, X_val, X_test, y_train, y_val, y_test)
            
            # Step 3: Model Training
            logger.info("\n" + "="*60)
            logger.info("STEP 3: MODEL TRAINING")
            logger.info("="*60)
            
            # Load preprocessors
            self.model_trainer.load_preprocessors()
            
            # Create pipelines
            self.model_trainer.build_model_pipelines()
            
            # Train models
            results = self.model_trainer.train_models(X_train, y_train, X_val, y_val, X_test, y_test)
            
            # Compare models
            comparison_df = self.model_trainer.compare_models()

            comparison_df.to_csv('reports/model_performance_comparison.csv', index_label='Model')
            
            # Plot results
            self.model_trainer.plot_model_comparison(comparison_df)
            self.model_trainer.plot_confusion_matrices(y_test)
            
            # Create comprehensive evaluation plots
            from utils import create_model_evaluation_plots, create_model_comparison_plots, comprehensive_shap_analysis
            
            # Create evaluation plots for best model
            best_model_name = self.model_trainer.best_model_name
            best_model = self.model_trainer.results[best_model_name]['pipeline']
            y_test_pred = self.model_trainer.results[best_model_name]['y_test_pred']
            y_test_proba = self.model_trainer.results[best_model_name]['y_test_proba']
            
            create_model_evaluation_plots(y_test, y_test_pred, y_test_proba, 
                                        self.model_trainer.label_encoder.classes_, 
                                        best_model_name)
            
            # Create model comparison plots
            create_model_comparison_plots(self.model_trainer.results)
            
            # Perform comprehensive SHAP analysis
            comprehensive_shap_analysis(best_model, X_train, X_val, X_test, y_test,
                                      X_train.columns.tolist(), 
                                      self.model_trainer.label_encoder.classes_,
                                      best_model_name)
            
            # Save best model
            self.model_trainer.save_best_model()
            
            # Pipeline summary
            logger.info("\n" + "="*80)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("="*80)
            
            pipeline_results = {
                'data_shape': data.shape,
                'validation_results': validation_results,
                'train_shape': X_train.shape,
                'val_shape': X_val.shape,
                'test_shape': X_test.shape,
                'best_model': self.model_trainer.best_model_name,
                'best_accuracy': self.model_trainer.results[self.model_trainer.best_model_name]['test_metrics']['accuracy'],
                'all_results': {name: {
                    'train_accuracy': result['train_metrics']['accuracy'],
                    'val_accuracy': result['val_metrics']['accuracy'],
                    'test_accuracy': result['test_metrics']['accuracy'],
                    'train_val_gap': result['train_metrics']['accuracy'] - result['val_metrics']['accuracy'],
                    'val_test_gap': result['val_metrics']['accuracy'] - result['test_metrics']['accuracy']
                } for name, result in self.model_trainer.results.items()}
            }
            
            logger.info("📊 PIPELINE SUMMARY:")
            logger.info(f"   • Data loaded: {pipeline_results['data_shape']}")
            logger.info(f"   • Train set: {pipeline_results['train_shape']}")
            logger.info(f"   • Validation set: {pipeline_results['val_shape']}")
            logger.info(f"   • Test set: {pipeline_results['test_shape']}")
            logger.info(f"   • Best model: {pipeline_results['best_model']}")
            logger.info(f"   • Best test accuracy: {pipeline_results['best_accuracy']:.4f}")
            logger.info(f"   • Models trained: {len(pipeline_results['all_results'])}")
            
            logger.info("\n✅ Training pipeline completed successfully!")
            logger.info("   All models saved and ready for deployment")
            
            return pipeline_results
            
        except Exception as e:
            logger.error(f"Pipeline failed with error: {str(e)}")
            raise
    
    def validate_pipeline(self) -> bool:
        """
        Validate that all pipeline components are working correctly
        """
        logger.info("Validating pipeline components...")
        
        try:
            # Test data ingestion
            data = self.data_ingestion.load_data()
            assert data is not None, "Data ingestion failed"
            logger.info("✅ Data ingestion validated")
            
            # Test data transformation
            clean_data = self.data_transformation.clean_data(data)
            features_data = self.data_transformation.engineer_features(clean_data)
            X_train, X_val, X_test, y_train, y_val, y_test = self.data_transformation.split_data(features_data)
            assert X_train is not None, "Data transformation failed"
            logger.info("✅ Data transformation validated")
            
            # Test model training setup
            self.model_trainer.load_preprocessors()
            pipelines = self.model_trainer.create_pipelines()
            assert len(pipelines) > 0, "Model training setup failed"
            logger.info("✅ Model training setup validated")
            
            logger.info("✅ All pipeline components validated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Pipeline validation failed: {str(e)}")
            return False

def main():
    """
    Main function to run the training pipeline
    """
    # Create and run pipeline
    pipeline = TrainingPipeline()
    
    # Validate pipeline first
    if not pipeline.validate_pipeline():
        logger.error("Pipeline validation failed. Exiting.")
        return
    
    # Run pipeline
    results = pipeline.run_pipeline()
    
    return results

if __name__ == "__main__":
    results = main()