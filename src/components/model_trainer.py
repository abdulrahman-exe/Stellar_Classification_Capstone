"""
Model Trainer Component
Trains multiple ML models and compares their performance
"""

import pandas as pd
import numpy as np
import logging
import joblib
from pathlib import Path
from typing import Dict, Tuple, Any
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix, 
    precision_score, recall_score, f1_score, roc_auc_score,
    precision_recall_curve, roc_curve, auc
)
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config import (
    MODEL_CONFIGS, SMOTE_CONFIG, RANDOM_STATE, 
    BEST_MODEL_FILE, LABEL_ENCODER_FILE, PREPROCESSOR_FILE,
    CUSTOM_PALETTE, CLASS_COLORS, PLOT_CONFIG, LOG_FORMAT
)

# Setup logging
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Handles model training, evaluation, and comparison
    """
    
    def __init__(self):
        self.models = {}
        self.model_pipelines = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.label_encoder = None
        
    def load_preprocessors(self):
        """
        Load preprocessor objects
        """
        logger.info("Loading preprocessors...")
        self.label_encoder = joblib.load(LABEL_ENCODER_FILE)
        logger.info("✅ Preprocessors loaded")
    
    def create_pipelines(self) -> Dict[str, ImbPipeline]:
        """
        Create model pipelines with Scaler → SMOTE → Classifier
        """
        logger.info("=" * 60)
        logger.info("CREATING MODEL PIPELINES")
        logger.info("=" * 60)
        
        pipelines = {}
        
        # Random Forest Pipeline
        logger.info("Creating Random Forest pipeline...")
        rf_pipeline = ImbPipeline([
            ('scaler', StandardScaler()),  # Will be set later
            ('smote', SMOTE(**SMOTE_CONFIG)),
            ('classifier', RandomForestClassifier(**MODEL_CONFIGS['RandomForest']))
        ])
        pipelines['Random Forest'] = rf_pipeline
        
        # XGBoost Pipeline
        logger.info("Creating XGBoost pipeline...")
        xgb_pipeline = ImbPipeline([
            ('scaler', StandardScaler()),  # Will be set later
            ('smote', SMOTE(**SMOTE_CONFIG)),
            ('classifier', XGBClassifier(**MODEL_CONFIGS['XGBoost']))
        ])
        pipelines['XGBoost'] = xgb_pipeline
        
        # Gradient Boosting Pipeline
        logger.info("Creating Gradient Boosting pipeline...")
        gb_pipeline = ImbPipeline([
            ('scaler', StandardScaler()),  # Will be set later
            ('smote', SMOTE(**SMOTE_CONFIG)),
            ('classifier', GradientBoostingClassifier(**MODEL_CONFIGS['GradientBoosting']))
        ])
        pipelines['Gradient Boosting'] = gb_pipeline
        
        # SVM Pipeline
        logger.info("Creating SVM pipeline (SGDClassifier)...")
        from sklearn.linear_model import SGDClassifier

        svm_pipeline = ImbPipeline([
            ('scaler', StandardScaler()),
            ('smote', SMOTE(**SMOTE_CONFIG)),
            ('classifier', SGDClassifier(
                loss='log_loss',
                max_iter=1000,
                random_state=42,
                n_jobs=-1,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10
            ))
        ])
        pipelines['SVM'] = svm_pipeline
        
        # Logistic Regression Pipeline
        logger.info("Creating Logistic Regression pipeline...")
        lr_pipeline = ImbPipeline([
            ('scaler', StandardScaler()),
            ('smote', SMOTE(**SMOTE_CONFIG)),
            ('classifier', LogisticRegression(**MODEL_CONFIGS['LogisticRegression']))
        ])
        pipelines['Logistic Regression'] = lr_pipeline
        
        self.model_pipelines = pipelines
        logger.info(f"✅ Created {len(pipelines)} model pipelines")
        
        return pipelines
    
    def _calculate_comprehensive_metrics(self, y_true: pd.Series, y_pred: np.ndarray, 
                                       y_proba: np.ndarray, set_name: str) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro'),
            'recall_macro': recall_score(y_true, y_pred, average='macro'),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted'),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted'),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted')
        }
        
        # Calculate ROC AUC for multiclass
        try:
            metrics['roc_auc_ovr'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
            metrics['roc_auc_ovo'] = roc_auc_score(y_true, y_proba, multi_class='ovo', average='macro')
        except:
            metrics['roc_auc_ovr'] = 0.0
            metrics['roc_auc_ovo'] = 0.0
        
        return metrics
    
    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series, 
                    X_val: pd.DataFrame, y_val: pd.Series,
                    X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Dict]:
        """
        Train all models and evaluate performance with comprehensive metrics
        Includes validation set evaluation
        """
        logger.info("=" * 60)
        logger.info("TRAINING MODELS WITH COMPREHENSIVE EVALUATION")
        logger.info("=" * 60)
        
        # Load scaler
        scaler = joblib.load(PREPROCESSOR_FILE)
        
        # Set scaler in all pipelines
        for pipeline in self.model_pipelines.values():
            pipeline.named_steps['scaler'] = scaler
        
        results = {}
        
        for model_name, pipeline in self.model_pipelines.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"TRAINING {model_name.upper()}")
            logger.info(f"{'='*60}")
            
            try:
                # Train model
                logger.info(f"🔄 Training {model_name} model...")
                pipeline.fit(X_train, y_train)
                logger.info("✅ Model trained successfully!")
                
                # Make predictions on all sets
                y_train_pred = pipeline.predict(X_train)
                y_val_pred = pipeline.predict(X_val)
                y_test_pred = pipeline.predict(X_test)
                
                # Get prediction probabilities
                y_train_proba = pipeline.predict_proba(X_train)
                y_val_proba = pipeline.predict_proba(X_val)
                y_test_proba = pipeline.predict_proba(X_test)
                
                # Calculate comprehensive metrics
                train_metrics = self._calculate_comprehensive_metrics(y_train, y_train_pred, y_train_proba, "Train")
                val_metrics = self._calculate_comprehensive_metrics(y_val, y_val_pred, y_val_proba, "Validation")
                test_metrics = self._calculate_comprehensive_metrics(y_test, y_test_pred, y_test_proba, "Test")
                
                logger.info(f"📊 Performance Metrics:")
                logger.info(f"   Train Accuracy: {train_metrics['accuracy']:.4f} ({train_metrics['accuracy']*100:.2f}%)")
                logger.info(f"   Val Accuracy:   {val_metrics['accuracy']:.4f} ({val_metrics['accuracy']*100:.2f}%)")
                logger.info(f"   Test Accuracy:  {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)")
                logger.info(f"   Train-Val Gap:  {(train_metrics['accuracy'] - val_metrics['accuracy'])*100:.2f}%")
                logger.info(f"   Val-Test Gap:   {(val_metrics['accuracy'] - test_metrics['accuracy'])*100:.2f}%")
                
                # Classification report for test set
                logger.info("📈 Classification Report (Test Set):")
                report = classification_report(y_test, y_test_pred, 
                                            target_names=self.label_encoder.classes_, 
                                            digits=4)
                logger.info(report)
                
                # Store results
                results[model_name] = {
                    'pipeline': pipeline,
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics,
                    'test_metrics': test_metrics,
                    'y_train_pred': y_train_pred,
                    'y_val_pred': y_val_pred,
                    'y_test_pred': y_test_pred,
                    'y_train_proba': y_train_proba,
                    'y_val_proba': y_val_proba,
                    'y_test_proba': y_test_proba,
                    'classification_report': report
                }
                
                # Store model
                self.models[model_name] = pipeline
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                continue
        
        self.results = results
        logger.info(f"\n✅ Training complete! {len(results)} models trained successfully")
        
        return results
    
    def compare_models(self) -> pd.DataFrame:
        """
        Compare all trained models and identify the best one
        Includes validation metrics for comprehensive evaluation
        """
        logger.info("=" * 60)
        logger.info("MODEL PERFORMANCE COMPARISON")
        logger.info("=" * 60)
        
        # Create comprehensive comparison dataframe
        comparison_data = []
        for model_name, result in self.results.items():
            train_metrics = result['train_metrics']
            val_metrics = result['val_metrics']
            test_metrics = result['test_metrics']
            
            comparison_data.append({
                'Model': model_name,
                'Train_Accuracy': train_metrics['accuracy'],
                'Val_Accuracy': val_metrics['accuracy'],
                'Test_Accuracy': test_metrics['accuracy'],
                'Train_Val_Gap': train_metrics['accuracy'] - val_metrics['accuracy'],
                'Val_Test_Gap': val_metrics['accuracy'] - test_metrics['accuracy'],
                'Val_F1_Macro': val_metrics['f1_macro'],
                'Test_F1_Macro': test_metrics['f1_macro'],
                'Val_ROC_AUC': val_metrics['roc_auc_ovr'],
                'Test_ROC_AUC': test_metrics['roc_auc_ovr']
            })
        
        comparison_df = pd.DataFrame(comparison_data).sort_values('Val_Accuracy', ascending=False)
        
        logger.info("\nComprehensive Model Performance Comparison:")
        logger.info(comparison_df.round(4).to_string(index=False))
        
        # Identify best model based on validation accuracy
        self.best_model_name = comparison_df.iloc[0]['Model']
        best_test_acc = comparison_df.iloc[0]['Test_Accuracy']
        
        logger.info(f"\n🏆 BEST MODEL: {self.best_model_name}")
        logger.info(f"   Test Accuracy: {best_test_acc:.4f} ({best_test_acc*100:.2f}%)")
        
        # Store best model
        self.best_model = self.results[self.best_model_name]['pipeline']
        
        return comparison_df
    
    def plot_model_comparison(self, comparison_df: pd.DataFrame) -> None:
        """
        Create visualization comparing model performance
        """
        logger.info("Creating model comparison visualization...")
        
        # Set plot style
        plt.rcParams['figure.dpi'] = PLOT_CONFIG['figure_dpi']
        sns.set(rc={'axes.facecolor': PLOT_CONFIG['axes_facecolor']}, style=PLOT_CONFIG['seaborn_style'])
        
        # Create comparison plot
        fig, ax = plt.subplots(figsize=(14, 8))
        x = np.arange(len(comparison_df))
        width = 0.25
        
        bars1 = ax.bar(x - width, comparison_df['Train_Accuracy'], width, 
                      label='Train Accuracy', color='#00BFFF', alpha=0.8)
        bars2 = ax.bar(x, comparison_df['Val_Accuracy'], width, 
                      label='Validation Accuracy', color='#9400D3', alpha=0.8)
        bars3 = ax.bar(x + width, comparison_df['Test_Accuracy'], width,
                      label='Test Accuracy', color='#FF6347', alpha=0.8)
        
        ax.set_xlabel('Model', fontweight='bold', fontsize=12)
        ax.set_ylabel('Accuracy', fontweight='bold', fontsize=12)
        ax.set_title('Model Accuracy Comparison - Stellar Classification', 
                    fontweight='bold', fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0.9, 1.0])
        
        # Add value labels on bars
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('reports/figures/model_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info("✅ Model comparison plot saved")
    
    def plot_confusion_matrices(self, y_test: pd.Series) -> None:
        """
        Plot confusion matrices for all models
        """
        logger.info("Creating confusion matrices...")
        
        n_models = len(self.results)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        colors = ['Blues', 'Greens', 'Oranges', 'Reds', 'Purples']
        
        for i, (model_name, result) in enumerate(self.results.items()):
            if i >= len(axes):
                break
                
            cm = confusion_matrix(y_test, result['y_test_pred'])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap=colors[i], 
                       xticklabels=self.label_encoder.classes_, 
                       yticklabels=self.label_encoder.classes_,
                       cbar_kws={'label': 'Count'}, ax=axes[i])
            axes[i].set_title(f'Confusion Matrix - {model_name}', 
                            fontsize=14, fontweight='bold')
            axes[i].set_xlabel('Predicted Class', fontweight='bold')
            axes[i].set_ylabel('Actual Class', fontweight='bold')
        
        # Hide unused subplots
        for i in range(len(self.results), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('reports/figures/confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info("✅ Confusion matrices saved")
    
    def save_best_model(self) -> None:
        """
        Save the best performing model
        """
        logger.info("=" * 60)
        logger.info("SAVING BEST MODEL")
        logger.info("=" * 60)
        
        if self.best_model is None:
            raise ValueError("No best model selected. Run compare_models() first.")
        
        # Save best model
        joblib.dump(self.best_model, BEST_MODEL_FILE)
        logger.info(f"💾 Best model saved as: {BEST_MODEL_FILE}")
        
        # Save model metadata
        metadata = {
            'best_model_name': self.best_model_name,
            'best_test_accuracy': self.results[self.best_model_name]['test_metrics']['accuracy'],
            'all_results': {name: {
                'train_accuracy': result['train_metrics']['accuracy'],
                'val_accuracy': result['val_metrics']['accuracy'],
                'test_accuracy': result['test_metrics']['accuracy'],
            } for name, result in self.results.items()}
        }

        joblib.dump(metadata, 'saved_models/model_metadata.pkl')
        logger.info("💾 Model metadata saved")

        logger.info(f"\n📊 Best Model Details:")
        logger.info(f"   Algorithm: {self.best_model_name}")
        logger.info(f"   Test Accuracy: {self.results[self.best_model_name]['test_metrics']['accuracy']:.4f}")
        logger.info(f"   Features: {len(self.best_model.named_steps['classifier'].feature_importances_) if hasattr(self.best_model.named_steps['classifier'], 'feature_importances_') else 'N/A'}")
        logger.info(f"   Classes: {len(self.label_encoder.classes_)}")
        logger.info(f"\n✅ Model ready for deployment!")
    
    def build_model_pipelines(self):
        """
        Compatibility wrapper so train_pipeline can always call `build_model_pipelines`.
        tries common existing method names, uses existing pipelines if present,
        or raises a clear error.
        """
        import logging
        logger = logging.getLogger(__name__)

        # If pipelines already created, nothing to do
        if getattr(self, "model_pipelines", None):
            logger.info("✅ Model pipelines already created (model_pipelines present).")
            return

        # Common alternative method names that may already exist in this class
        alternatives = (
            "create_model_pipelines",
            "create_pipelines",
            "build_pipelines",
            "setup_pipelines",
            "initialize_pipelines",
        )
        for name in alternatives:
            if hasattr(self, name):
                logger.info(f"Calling existing pipeline builder '{name}'...")
                getattr(self, name)()
                # ensure attribute is set for future calls
                if not getattr(self, "model_pipelines", None) and getattr(self, "pipelines", None):
                    self.model_pipelines = getattr(self, "pipelines")
                logger.info("✅ Model pipelines created.")
                return

        # Fallback: maybe pipelines were created during __init__ under a different attribute
        if getattr(self, "pipelines", None):
            self.model_pipelines = getattr(self, "pipelines")
            logger.info("Using existing 'pipelines' attribute as model_pipelines.")
            return

        raise AttributeError(
            "ModelTrainer: no pipeline creation method found. "
            "Implement `build_model_pipelines()` or one of: "
            "create_model_pipelines/create_pipelines/build_pipelines/setup_pipelines."
        )

def main():
    """
    Main function for testing model training
    """
    from data_transformation import DataTransformation
    
    # Load preprocessed data
    transformer = DataTransformation()
    transformer.load_preprocessors()
    
    # Load train/test data
    X_train = pd.read_csv('data/processed/train.csv').drop('class', axis=1)
    y_train = pd.read_csv('data/processed/train.csv')['class']
    X_test = pd.read_csv('data/processed/test.csv').drop('class', axis=1)
    y_test = pd.read_csv('data/processed/test.csv')['class']
    
    # Encode target
    y_train_encoded = transformer.label_encoder.transform(y_train)
    y_test_encoded = transformer.label_encoder.transform(y_test)
    
    # Train models
    trainer = ModelTrainer()
    trainer.load_preprocessors()
    trainer.create_pipelines()
    results = trainer.train_models(X_train, y_train_encoded, X_test, y_test_encoded)
    
    # Compare models
    comparison_df = trainer.compare_models()
    
    # Plot results
    trainer.plot_model_comparison(comparison_df)
    trainer.plot_confusion_matrices(y_test_encoded)
    
    # Save best model
    trainer.save_best_model()
    
    return trainer, results, comparison_df

if __name__ == "__main__":
    trainer, results, comparison_df = main()