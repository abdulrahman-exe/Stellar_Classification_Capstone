"""
Utility functions for SHAP analysis, plotting, and logging
"""

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from pathlib import Path
from typing import List, Any
import sys
import warnings
import json
import traceback
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize

warnings.filterwarnings('ignore')


sys.path.append(str(Path(__file__).parent))

from config import (
    CUSTOM_PALETTE, CLASS_COLORS, PLOT_CONFIG, SHAP_CONFIG,
    LOG_FORMAT, BEST_MODEL_FILE, PREPROCESSOR_FILE, LABEL_ENCODER_FILE
)

# Setup logging
# This configures the root logger.
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT,
                    handlers=[
                        logging.FileHandler("stellar_classification.log"),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

def setup_plotting():
    """
    Setup plotting configuration
    """
    # Setting the resolution of the plotted figures
    plt.rcParams['figure.dpi'] = PLOT_CONFIG['figure_dpi']
    
    # Configure Seaborn plot styles: Set background color and use dark grid
    sns.set(rc={'axes.facecolor': PLOT_CONFIG['axes_facecolor']}, 
            style=PLOT_CONFIG['seaborn_style'])

def create_shap_explainer(model, X_sample: np.ndarray) -> shap.Explainer:
    """
    Create SHAP explainer for the model
    """
    logger.info("Creating SHAP explainer...")
    
    # Get the trained model from the pipeline
    trained_model = model.named_steps['classifier']
    
    # Create SHAP explainer based on model type
    model_name = type(trained_model).__name__
    
    if model_name in ['RandomForestClassifier', 'XGBClassifier', 'GradientBoostingClassifier']:
        explainer = shap.TreeExplainer(trained_model)
        logger.info("✅ Using TreeExplainer for tree-based model")
    else:
        logger.warning(f"Using KernelExplainer for {model_name}. This can be slow.")
        # Sample the data for the explainer background
        X_sample_summary = shap.sample(X_sample, min(50, X_sample.shape[0]))
        explainer = shap.KernelExplainer(model.predict_proba, X_sample_summary)
        logger.info(f"✅ Using KernelExplainer for {model_name}")
    
    return explainer

def calculate_shap_values(explainer: shap.Explainer, X_sample: np.ndarray) -> List[np.ndarray]:
    """
    Calculate SHAP values for the sample data
    """
    logger.info("Calculating SHAP values...")
    
    # Calculate SHAP values for a sample of data
    sample_size = min(SHAP_CONFIG['sample_size'], len(X_sample))
    X_sample_subset = X_sample[:sample_size]
    
    # Get SHAP values
    shap_values = explainer.shap_values(X_sample_subset)
    
    logger.info(f"✅ SHAP values calculated for {sample_size} samples")
    logger.info(f"   Shape: {np.array(shap_values).shape}")
    
    # For multi-class models, SHAP returns values for each class
    if isinstance(shap_values, list):
        logger.info(f"   Multi-class model: {len(shap_values)} classes")
    else:
        logger.info("   Single output model")
    
    return shap_values, X_sample_subset

def plot_feature_importance(shap_values: List[np.ndarray], feature_names: List[str], 
                            model_name: str) -> pd.DataFrame:
    """
    Plot global feature importance from SHAP values
    """
    logger.info("Creating feature importance plot...")
    
    # Calculate mean absolute SHAP values for global importance
    if isinstance(shap_values, list):
        # For multi-class, average across all classes
        mean_shap_values = np.mean([np.abs(shap_vals) for shap_vals in shap_values], axis=0)
        feature_importance = np.mean(mean_shap_values, axis=0)
    else:
        # For single output
        feature_importance = np.mean(np.abs(shap_values), axis=0)
    
    # Create feature importance dataframe
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)
    
    logger.info("📊 Top 10 Most Important Features:")
    logger.info(feature_importance_df.head(10).to_string(index=False))
    
    # Visualize global feature importance
    fig, ax = plt.subplots(figsize=(12, 8)) # Use fig, ax
    top_features = feature_importance_df.head(15)
    bars = ax.barh(range(len(top_features)), top_features['Importance'], 
                       color='#9400D3', alpha=0.8)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['Feature'])
    ax.set_xlabel('Mean |SHAP Value|', fontweight='bold', fontsize=12)
    ax.set_title(f'Global Feature Importance - {model_name}', 
                 fontweight='bold', fontsize=16)
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis() # Show most important at top
    
    # Add value labels on bars
    for bar in bars:
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', ha='left', va='center', fontweight='bold')
    
    plt.tight_layout()
    Path('reports/figures').mkdir(parents=True, exist_ok=True) # Ensure dir exists
    plt.savefig('reports/figures/shap_feature_importance.png', dpi=300, bbox_inches='tight')
    # plt.show() # Commented out for automated pipeline
    plt.close(fig) # Close figure to free memory
    
    logger.info("✅ Feature importance plot saved")
    
    return feature_importance_df

def plot_shap_summary(shap_values: List[np.ndarray], X_sample: np.ndarray, 
                      feature_names: List[str], class_names: List[str]) -> None:
    """
    Create SHAP summary plot
    """
    logger.info("Creating SHAP summary plot...")
    
    # Ensure X_sample is a DataFrame for feature names
    if not isinstance(X_sample, pd.DataFrame):
        X_sample_df = pd.DataFrame(X_sample, columns=feature_names)
    else:
        X_sample_df = X_sample

    if isinstance(shap_values, list):
        # For multi-class, plot all classes on one summary plot
        shap.summary_plot(shap_values, X_sample_df, 
                          class_names=class_names,
                          show=False, max_display=15)
        fig = plt.gcf() # Get current figure
        fig.suptitle('SHAP Summary Plot (All Classes)', fontweight='bold', fontsize=16, y=1.02)
    else:
        # For single output
        shap.summary_plot(shap_values, X_sample_df,
                          show=False, max_display=15)
        fig = plt.gcf() # Get current figure
        plt.title('SHAP Summary Plot', fontweight='bold', fontsize=16)

    plt.tight_layout()
    # plt.show() # Commented out for automated pipeline
    plt.close(fig) # Close figure
    
    logger.info("✅ SHAP summary plot created")

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                          class_names: List[str], model_name: str) -> None:
    """
    Plot confusion matrix
    """
    logger.info(f"Creating confusion matrix for {model_name}...")
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6)) # Use fig, ax
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'}, ax=ax)
    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=16, fontweight='bold')
    ax.set_xlabel('Predicted Class', fontweight='bold')
    ax.set_ylabel('Actual Class', fontweight='bold')
    
    plt.tight_layout()
    Path('reports/figures').mkdir(parents=True, exist_ok=True) # Ensure dir exists
    plt.savefig(f'reports/figures/{model_name.lower().replace(" ", "_")}_confusion_matrix.png', 
                dpi=300, bbox_inches='tight')
    # plt.show() # Commented out for automated pipeline
    plt.close(fig) # Close figure
    
    logger.info("✅ Confusion matrix saved")

def plot_class_distribution(y: np.ndarray, class_names: List[str], 
                            title: str = "Class Distribution") -> None:
    """
    Plot class distribution
    """
    logger.info("Creating class distribution plot...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Count plot
    unique, counts = np.unique(y, return_counts=True)
    
    # Ensure class_names match the order of unique labels
    # This is a robust fix in case y labels are not 0, 1, 2...
    label_map = {label: name for label, name in enumerate(class_names)}
    ordered_names = [label_map[label] for label in unique]
    
    bars = axes[0].bar(ordered_names, counts, color=CUSTOM_PALETTE, alpha=0.8)
    axes[0].set_title('Class Distribution (Count)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Count', fontweight='bold')
    axes[0].set_xlabel('Stellar Object Class', fontweight='bold')
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                     f'{count}', ha='center', va='bottom', fontweight='bold')
    
    # Pie chart
    axes[1].pie(counts, labels=ordered_names, autopct='%1.1f%%', 
                colors=CUSTOM_PALETTE, startangle=90, 
                textprops={'fontsize': 12, 'weight': 'bold'})
    axes[1].set_title('Class Distribution (Percentage)', fontsize=14, fontweight='bold')
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    Path('reports/figures').mkdir(parents=True, exist_ok=True) # Ensure dir exists
    plt.savefig('reports/figures/class_distribution.png', dpi=300, bbox_inches='tight')
    # plt.show() # Commented out for automated pipeline
    plt.close(fig) # Close figure
    
    logger.info("✅ Class distribution plot saved")

def analyze_feature_interactions(shap_values: List[np.ndarray], X_sample: np.ndarray, 
                                 feature_names: List[str]) -> None:
    """
    Analyze feature interactions using SHAP
    """
    if len(feature_names) < 2:
        logger.warning("Not enough features to analyze interaction. Skipping.")
        return
        
    logger.info("Analyzing feature interactions...")
    
    # For multi-class, use the first class for interaction analysis
    if isinstance(shap_values, list):
        interaction_shap = shap_values[0]  # Use first class
    else:
        interaction_shap = shap_values
        
    # Ensure X_sample is a DataFrame for feature names
    if not isinstance(X_sample, pd.DataFrame):
        X_sample_df = pd.DataFrame(X_sample, columns=feature_names)
    else:
        X_sample_df = X_sample

    # Create interaction plot
    fig = plt.figure(figsize=(12, 8))
    # Using 'redshift' as an example if available, otherwise default
    interaction_feat = 'redshift' if 'redshift' in feature_names else feature_names[1]
    
    shap.dependence_plot(feature_names[0], interaction_shap, X_sample_df, 
                         interaction_index=interaction_feat, show=False)
                         
    plt.title(f'Feature Interaction: {feature_names[0]} vs {interaction_feat}', 
              fontweight='bold', fontsize=14)
    plt.tight_layout()
    Path('reports/figures').mkdir(parents=True, exist_ok=True) # Ensure dir exists
    plt.savefig('reports/figures/feature_interaction.png', dpi=300, bbox_inches='tight')
    # plt.show() # Commented out for automated pipeline
    plt.close(fig) # Close figure
    
    logger.info("✅ Feature interaction plot saved")

def create_waterfall_plot(explainer: shap.Explainer, shap_values: List[np.ndarray],
                          X_sample: np.ndarray, example_idx: int, 
                          class_name: str) -> None:
    """
    Create SHAP waterfall plot for a specific example
    """
    logger.info(f"Creating waterfall plot for {class_name} example...")
    
    # Handle multi-class vs single-class
    if isinstance(shap_values, list):
        # Multi-class: Need to get the correct class index
        class_idx = explainer.classes_.tolist().index(class_name)
        shap_vals_for_class = shap_values[class_idx][example_idx]
        expected_value = explainer.expected_value[class_idx]
    else:
        shap_vals_for_class = shap_values[example_idx]
        expected_value = explainer.expected_value

    # Create waterfall plot
    # We must use Explanation objects for the new SHAP API
    shap_explanation = shap.Explanation(
        values=shap_vals_for_class,
        base_values=expected_value,
        data=X_sample[example_idx],
        feature_names=X_sample.columns.tolist() if isinstance(X_sample, pd.DataFrame) else None
    )

    fig = plt.figure() # Create figure to save
    shap.plots.waterfall(shap_explanation, max_display=10, show=False)
    plt.title(f'{class_name} Example (Index {example_idx})', fontweight='bold', fontsize=12)
    plt.tight_layout()
    
    Path('reports/figures').mkdir(parents=True, exist_ok=True) # Ensure dir exists
    plt.savefig(f'reports/figures/waterfall_{class_name.lower()}.png', 
                dpi=300, bbox_inches='tight')
    # plt.show() # Commented out for automated pipeline
    plt.close(fig) # Close figure
    
    logger.info("✅ Waterfall plot saved")

def save_shap_analysis_results(feature_importance_df: pd.DataFrame, 
                               model_name: str) -> None:
    """
    Save SHAP analysis results
    """
    logger.info("Saving SHAP analysis results...")
    Path('reports').mkdir(parents=True, exist_ok=True) # Ensure dir exists
    
    # Save feature importance
    feature_importance_df.to_csv('reports/shap_feature_importance.csv', index=False)
    
    # Save summary
    summary = {
        'model_name': model_name,
        'top_features': feature_importance_df.head(10).to_dict('records'),
        'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open('reports/shap_analysis_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("✅ SHAP analysis results saved")

# Removed redundant setup_logging function. The one at the top is used.

def log_model_performance(model_name: str, train_acc: float, test_acc: float, 
                          overfitting_gap: float) -> None:
    """
    Log model performance metrics
    """
    logger.info(f"📊 {model_name} Performance:")
    logger.info(f"   Train Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
    logger.info(f"   Test Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
    logger.info(f"   Overfitting Gap: {overfitting_gap*100:.2f}%")

def comprehensive_shap_analysis(model, X_train, X_val, X_test, y_test, 
                              feature_names, class_names, model_name="Model", sample_size=None):
    """
    Perform comprehensive SHAP analysis with all visualizations
    Fixed to work with imblearn pipelines
    sample_size: int or None, number of samples to use for SHAP value calculation (default from SHAP_CONFIG)
    """
    logger.info("=" * 60)
    logger.info("COMPREHENSIVE SHAP ANALYSIS")
    logger.info("=" * 60)
    
    # CRITICAL FIX: the entire function was wrapped in a try block
    # without a corresponding 'except' block.
    try:
        # Extract classifier from pipeline
        if hasattr(model, 'named_steps'):
            classifier = model.named_steps['classifier']
            # FIX: Use .get() to avoid KeyError if 'scaler' doesn't exist
            scaler = model.named_steps.get('scaler', None)
            
            # Use X_train for background, X_val for analysis
            if scaler is not None:
                X_train_scaled = scaler.transform(X_train)
                X_val_scaled = scaler.transform(X_val)
            else:
                X_train_scaled = X_train
                X_val_scaled = X_val
        else:
            classifier = model
            X_train_scaled = X_train
            X_val_scaled = X_val
        
        # Convert to DataFrame for SHAP plots
        X_val_scaled_df = pd.DataFrame(X_val_scaled, columns=feature_names)

        logger.info(f"Model type: {type(classifier).__name__}")
        
        # Initialize SHAP explainer based on model type
        model_name_type = type(classifier).__name__
        
        logger.info("Initializing SHAP explainer...")
        
        # Use provided sample_size or default from config
        if sample_size is None:
            sample_size = min(SHAP_CONFIG.get('sample_size', 100), len(X_val_scaled))
        else:
            sample_size = min(sample_size, len(X_val_scaled))
        
        # Use a background dataset for KernelExplainer
        # Using X_train (or a sample of it) is standard practice
        background_sample_size = min(100, len(X_train_scaled))
        background_data = shap.sample(X_train_scaled, background_sample_size)
        
        try:
            if model_name_type in ['RandomForestClassifier', 'XGBClassifier', 'GradientBoostingClassifier']:
                explainer = shap.TreeExplainer(classifier)
                logger.info(f"✅ Using TreeExplainer for {model_name_type}")
            else:
                # For linear models, use KernelExplainer
                explainer = shap.KernelExplainer(
                    lambda x: classifier.predict_proba(x),
                    background_data
                )
                logger.info(f"✅ Using KernelExplainer for {model_name_type}")
        except Exception as e:
            logger.warning(f"Explainer initialization failed: {e}, using KernelExplainer...")
            explainer = shap.KernelExplainer(
                lambda x: classifier.predict_proba(x),
                background_data
            )
        
        # Calculate SHAP values for validation set (sample for performance)
        logger.info(f"Calculating SHAP values for {sample_size} samples...")
        # Use the DataFrame version for calculation
        X_sample_df = X_val_scaled_df.iloc[:sample_size]
        shap_values = explainer.shap_values(X_sample_df)
        logger.info("✅ SHAP values calculated")
        
        # 1. Global Feature Importance (Summary Plot)
        logger.info("Creating global feature importance plot (summary_plot)...")
        try:
            shap.summary_plot(shap_values, X_sample_df, 
                              class_names=class_names,
                              show=False, max_display=15)
            fig = plt.gcf()
            plt.title(f'SHAP Global Feature Importance - {model_name}', 
                      fontsize=16, fontweight='bold')
            plt.tight_layout()
            Path('reports/figures').mkdir(parents=True, exist_ok=True)
            plt.savefig('reports/figures/shap_global_importance.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
            logger.info("✅ Global importance plot saved")
        except Exception as e:
            logger.warning(f"Could not create summary plot: {e}")
            plt.close()
        
        # 2. SHAP Force Plot (for the first example)
        logger.info("Creating SHAP force plot...")
        try:
            # FIX: Simplified logic for force_plot
            if isinstance(shap_values, list):
                # Multi-class: use first class
                expected_value = explainer.expected_value[0]
                shap_vals_for_plot = shap_values[0][0] # First class, first instance
            else:
                # Single output
                expected_value = explainer.expected_value
                shap_vals_for_plot = shap_values[0] # First instance
            
            # Create a full Explanation object for the single instance
            force_explanation = shap.Explanation(
                values=shap_vals_for_plot,
                base_values=expected_value,
                data=X_sample_df.iloc[0],
                feature_names=feature_names
            )

            # Need to save the HTML plot from shap.force_plot
            html_plot = shap.force_plot(
                base_value=force_explanation.base_values,
                shap_values=force_explanation.values,
                features=force_explanation.data,
                feature_names=feature_names,
                show=False
            )
            Path('reports/figures').mkdir(parents=True, exist_ok=True)
            shap.save_html('reports/figures/shap_force_plot.html', html_plot)
            logger.info("✅ Force plot saved as HTML")
        except Exception as e:
            logger.warning(f"Could not create force plot: {e}")
            plt.close()  # Make sure to close any dangling plt figures
        
        # 3. Feature Importance Bar Plot
        # FIX: Replaced redundant code with a call to the existing function
        logger.info("Creating feature importance bar plot...")
        try:
            importance_df = plot_feature_importance(shap_values, feature_names, model_name)
            # Save the results
            save_shap_analysis_results(importance_df, model_name)
        except Exception as e:
            logger.warning(f"Could not create feature importance bar plot: {e}")
            plt.close()

    # CRITICAL FIX: Added the missing 'except' block to close the 'try'
    except Exception as e:
        logger.error(f"❌ Comprehensive SHAP analysis failed: {e}")
        logger.error(traceback.format_exc())


def create_shap_report(shap_values, X_val, feature_names, class_names, model_name):
    """
    Create a comprehensive SHAP analysis report
    """
    logger.info("Creating SHAP analysis report...")
    Path('reports').mkdir(parents=True, exist_ok=True) # Ensure dir exists
    
    # Calculate feature importance
    if isinstance(shap_values, list):
        # Multi-class: average across classes
        mean_shap_values = np.mean([np.abs(sv) for sv in shap_values], axis=0)
    else:
        mean_shap_values = np.abs(shap_values)
    
    # Check for shape mismatch before proceeding
    if mean_shap_values.shape[1] != len(feature_names):
         logger.error(f"Shape mismatch in create_shap_report: {mean_shap_values.shape[1]} values vs {len(feature_names)} features")
         return

    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': np.mean(mean_shap_values, axis=0)
    }).sort_values('importance', ascending=False)
    
    # Save feature importance
    feature_importance.to_csv('reports/shap_feature_importance.csv', index=False)
    
    # Create summary statistics
    summary_stats = {
        'model_name': model_name,
        'n_features': len(feature_names),
        'n_samples': len(X_val),
        'n_classes': len(class_names),
        'top_5_features': feature_importance.head(5)['feature'].tolist(),
        'mean_importance': feature_importance['importance'].mean(),
        'std_importance': feature_importance['importance'].std()
    }
    
    # Save summary
    summary_df = pd.DataFrame([summary_stats])
    summary_df.to_csv('reports/shap_analysis_summary.csv', index=False)
    
    logger.info("✅ SHAP report created successfully!")

def create_model_evaluation_plots(y_true, y_pred, y_proba, class_names, model_name, save_dir='reports/figures'):
    """
    Create comprehensive model evaluation plots
    """
    logger.info(f"Creating evaluation plots for {model_name}...")
    
    # Create save directory
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    fig_cm, ax_cm = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax_cm)
    ax_cm.set_title(f'Confusion Matrix - {model_name}', fontsize=16, fontweight='bold')
    ax_cm.set_xlabel('Predicted', fontsize=12)
    ax_cm.set_ylabel('Actual', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/confusion_matrix_{model_name.lower().replace(" ", "_")}.png', 
                dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close(fig_cm)
    
    # 2. ROC Curves for each class
    fig_roc, ax_roc = plt.subplots(figsize=(12, 8))
    y_bin = label_binarize(y_true, classes=range(len(class_names)))
    
    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.3f})', linewidth=2)
    
    ax_roc.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel('False Positive Rate', fontsize=12)
    ax_roc.set_ylabel('True Positive Rate', fontsize=12)
    ax_roc.set_title(f'ROC Curves - {model_name}', fontsize=16, fontweight='bold')
    ax_roc.legend(loc="lower right")
    ax_roc.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/roc_curves_{model_name.lower().replace(" ", "_")}.png', 
                dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close(fig_roc)
    
    # 3. Precision-Recall Curves
    fig_pr, ax_pr = plt.subplots(figsize=(12, 8))
    
    for i, class_name in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(y_bin[:, i], y_proba[:, i])
        avg_precision = average_precision_score(y_bin[:, i], y_proba[:, i])
        ax_pr.plot(recall, precision, label=f'{class_name} (AP = {avg_precision:.3f})', linewidth=2)
    
    ax_pr.set_xlim([0.0, 1.0])
    ax_pr.set_ylim([0.0, 1.05])
    ax_pr.set_xlabel('Recall', fontsize=12)
    ax_pr.set_ylabel('Precision', fontsize=12)
    ax_pr.set_title(f'Precision-Recall Curves - {model_name}', fontsize=16, fontweight='bold')
    ax_pr.legend(loc="lower left")
    ax_pr.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/pr_curves_{model_name.lower().replace(" ", "_")}.png', 
                dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close(fig_pr)
    
    logger.info(f"✅ Evaluation plots created for {model_name}")

def create_model_comparison_plots(model_results, save_dir='reports/figures'):
    """
    Create comprehensive model comparison visualizations
    """
    logger.info("Creating model comparison plots...")
    
    # Create save directory
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Extract metrics
    models = list(model_results.keys())
    if not models:
        logger.warning("No model results found to compare. Skipping plots.")
        return
        
    train_acc = [model_results[m]['train_metrics']['accuracy'] for m in models]
    val_acc = [model_results[m]['val_metrics']['accuracy'] for m in models]
    test_acc = [model_results[m]['test_metrics']['accuracy'] for m in models]
    
    # 1. Accuracy Comparison
    fig1, ax1 = plt.subplots(figsize=(14, 8))
    x = np.arange(len(models))
    width = 0.25
    
    bars1 = ax1.bar(x - width, train_acc, width, label='Train', color='#00BFFF', alpha=0.8)
    bars2 = ax1.bar(x, val_acc, width, label='Validation', color='#9400D3', alpha=0.8)
    bars3 = ax1.bar(x + width, test_acc, width, label='Test', color='#FF6347', alpha=0.8)
    
    ax1.set_xlabel('Model', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Accuracy', fontweight='bold', fontsize=12)
    ax1.set_title('Model Performance Comparison', fontweight='bold', fontsize=16)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/model_comparison_accuracy.png', dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close(fig1)
    
    # 2. F1 Score Comparison
    train_f1 = [model_results[m]['train_metrics']['f1_macro'] for m in models]
    val_f1 = [model_results[m]['val_metrics']['f1_macro'] for m in models]
    test_f1 = [model_results[m]['test_metrics']['f1_macro'] for m in models]
    
    fig2, ax2 = plt.subplots(figsize=(14, 8))
    bars1 = ax2.bar(x - width, train_f1, width, label='Train', color='#00BFFF', alpha=0.8)
    bars2 = ax2.bar(x, val_f1, width, label='Validation', color='#9400D3', alpha=0.8)
    bars3 = ax2.bar(x + width, test_f1, width, label='Test', color='#FF6347', alpha=0.8)
    
    ax2.set_xlabel('Model', fontweight='bold', fontsize=12)
    ax2.set_ylabel('F1 Score (Macro)', fontweight='bold', fontsize=12)
    ax2.set_title('Model F1 Score Comparison', fontweight='bold', fontsize=16)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/model_comparison_f1.png', dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close(fig2)
    
    logger.info("✅ Model comparison plots created")

def main():
    """
    Main function for testing utilities
    """
    # Setup plotting
    setup_plotting()
    
    # Logging is already set up at the top level
    
    logger.info("Utility functions loaded successfully")

if __name__ == "__main__":
    main()