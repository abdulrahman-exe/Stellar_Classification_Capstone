"""
Stellar Classification Streamlit App
Enhanced UI for a comprehensive web application
"""

import traceback
import streamlit as st
import base64
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap
import joblib
from pathlib import Path
import sys
import logging
import warnings

warnings.filterwarnings('ignore')

# --- Configuration & Setup ---

# Add src to path for imports
# Consider using editable installs (pip install -e .) for better path management
sys.path.append(str(Path(__file__).parent / "src"))

try:
    from src.pipelines.predict_pipeline import PredictionPipeline
    # Import config variables if needed elsewhere, otherwise remove if unused in app.py
    # from src.config import CUSTOM_PALETTE, CLASS_COLORS, ALL_FEATURES, TARGET_COLUMN
except ImportError as e:
    st.error(f"Failed to import necessary modules from src/: {e}")
    st.error("Ensure 'src/' directory is in the correct path and contains required files.")
    st.stop()


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Page Config & Styling ---

st.set_page_config(
    page_title="Stellar Classify Pro",
    page_icon="🔭", # Telescope icon
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "# Stellar Classification System\nBuilt using Streamlit, Scikit-learn, SHAP, and Plotly."
    }
)

# --- Function to Encode GIF ---
@st.cache_data
def get_img_as_base64(file_path):
    """Reads a binary file and returns its base64 encoded string."""
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        st.error(f"Background image file not found at: {file_path}")
        return None
    except Exception as e:
        st.error(f"Error reading background image: {e}")
        return None

# --- Add Background CSS ---

# 1. Define the path to your GIF
img_path = Path("assets/Gamma-ray_Burst.gif")

# 2. Encode the GIF
bg_image_base64 = get_img_as_base64(img_path)

# 3. Inject into CSS (if encoding was successful)
if bg_image_base64:
    # Use one of the CSS options (e.g., targeting body or .stApp)
    page_bg_img = f"""
    <style>
    .stApp {{ /* Target the main Streamlit app container */
    background-image: url("data:image/gif;base64,{bg_image_base64}");
    background-size: cover;
    background-position: center center;
    background-repeat: no-repeat;
    background-attachment: fixed;
    }}
    /* Optional transparency for content blocks */
    div[data-testid="stVerticalBlock"] > div[style*="flex-direction: column;"] > div[data-testid="stVerticalBlock"] {{
        background-color: rgba(255, 255, 255, 0.9); /* White with 90% opacity */
        border-radius: 8px; /* Add slight rounding */
    }}
    /* Optional: Make sidebar slightly transparent */
    [data-testid="stSidebar"] > div:first-child {{
    background-color: rgba(240, 242, 246, 0.85); /* Adjust transparency */
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)
else:
    st.warning("Could not load background image.")

# --- Enhanced CSS Styling ---
st.markdown("""
<style>
    /* Base theme colors */
    :root {
        --primary-color: #6a0dad; /* Deep Purple */
        --secondary-color: #00c6ff; /* Bright Blue */
        --accent-color: #ffde59; /* Gold */
        --background-color: #f0f2f6; /* Soft gray */
        --text-color: #333;
        --card-bg-color: #ffffff;
        --card-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }

    /* Apply background to main area */
    .main > div {
        /* background-color: var(--background-color); */ /* Optional: Full background color */
    }

/* --- Translucent Content Blocks --- */
div[data-testid="stVerticalBlock"] > div[style*="flex-direction: column;"] > div[data-testid="stVerticalBlock"] {{
    background-color: rgba(255, 255, 255, 0.9); /* White 90% opaque */
    padding: 1.5rem;
    border-radius: 8px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    margin-top: 1rem;
    transition: background-color 0.3s ease;
}}

/* --- Translucent Sidebar --- */
[data-testid="stSidebar"] > div:first-child {
    background-color: rgba(240, 242, 246, 0.85); /* Adjust opacity as needed */
    border-radius: 0 8px 8px 0; /* Add this line */ /* Top-left, Top-right, Bottom-right, Bottom-left */
}

/* Header */
.main-header {
    font-size: 3rem; /* Slightly smaller */
    font-weight: 700;
    text-align: center;
    margin-bottom: 1.5rem;
        padding-top: 1rem;
        background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Subtitle */
    .subtitle {
        text-align: center;
        font-size: 1.1rem;
        color: #555;
        margin-bottom: 2.5rem;
    }

    /* Prediction Card */
    .prediction-card {
        background: linear-gradient(135deg, var(--primary-color) 0%, #4a0a77 100%); /* Purple gradient */
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: var(--card-shadow);
    }
    .prediction-card h2 {
        font-size: 1.8rem;
        margin-bottom: 0.5rem;
    }
    .prediction-card h3 {
        font-size: 1.3rem;
        font-weight: 500;
        opacity: 0.9;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 5px; /* Spacing between tabs */
        border-bottom: 2px solid #ddd; /* Underline for tab list */
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent; /* Cleaner look */
        border-radius: 6px 6px 0 0; /* Slightly rounded top corners */
        border: none;
        border-bottom: 2px solid transparent; /* Prepare for active border */
        padding: 0 25px;
        margin-bottom: -2px; /* Overlap border */
        color: #555; /* Default tab text color */
        transition: all 0.2s ease-in-out;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e9eef2; /* Light hover effect */
        color: var(--primary-color);
    }
    .stTabs [aria-selected="true"] {
        background-color: transparent;
        color: var(--primary-color);
        font-weight: 600;
        border-bottom: 2px solid var(--primary-color); /* Active tab underline */
    }

    /* Section Styling */
    div[data-testid="stVerticalBlock"] > div[style*="flex-direction: column;"] > div[data-testid="stVerticalBlock"] {
         /* Targets containers within tabs */
        background-color: var(--card-bg-color);
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: var(--card-shadow);
        margin-top: 1rem;
    }

    /* Input Form Styling */
    div.stSlider > div[data-baseweb="slider"] {
        /* Style sliders if needed */
        color: var(--primary-color);
    }

    /* Button Styling */
    .stButton > button {
        background-color: var(--primary-color);
        color: white;
        border: none;
        padding: 0.7rem 1.5rem;
        border-radius: 8px;
        font-weight: 600;
        transition: background-color 0.2s ease;
    }
    .stButton > button:hover {
        background-color: #4a0a77; /* Darker purple on hover */
    }
    .stButton > button:active {
        background-color: #3a085e; /* Even darker when clicked */
    }

    /* Metrics Styling */
    div[data-testid="stMetric"] {
        background-color: #e9eef2; /* Light background for metrics */
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.9rem;
        color: #555;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--primary-color);
    }

</style>
""", unsafe_allow_html=True)


# --- Data and Model Loading Functions ---

@st.cache_resource # Use cache_resource for non-serializable objects like models
def load_model_artifacts():
    """Load model pipeline and associated artifacts."""
    try:
        pipeline_obj = PredictionPipeline()
        pipeline_obj.load_model() # Loads artifacts internally
        return pipeline_obj
    except FileNotFoundError as e:
        st.error(f"❌ Critical Error: Artifact file not found: {e}. Cannot load model.")
        st.error("   Ensure the training pipeline (`run_pipeline.py`) ran successfully and saved files exist in 'saved_models/'.")
        logger.error(f"Artifact loading failed: FileNotFoundError - {e}\n{traceback.format_exc()}")
        return None # Return None to indicate failure
    except Exception as e:
        st.error(f"❌ Critical Error: Failed to load model artifacts: {e}")
        logger.error(f"Artifact loading failed: {e}\n{traceback.format_exc()}")
        return None # Return None to indicate failure

@st.cache_data # Use cache_data for serializable data like DataFrames
def load_sample_data(_pipeline_obj: PredictionPipeline):
    """Load sample test data and ensure 'class' is integer."""
    if _pipeline_obj is None or _pipeline_obj.label_encoder is None:
        st.warning("Pipeline not loaded, cannot map class names for sample data.")
        return None

    try:
        data_path = Path("data/processed/test.csv")
        if data_path.exists():
            df = pd.read_csv(data_path)
            # Ensure 'class' column exists and is integer
            if 'class' in df.columns:
                 df['class'] = df['class'].astype(int)
                 # Map numeric class labels back to names for display
                 class_names_map = {idx: name for idx, name in enumerate(_pipeline_obj.label_encoder.classes_)}
                 df['class_name'] = df['class'].map(class_names_map)
            else:
                 st.warning("Sample data (test.csv) is missing the 'class' column.")
                 df['class_name'] = 'Unknown' # Add placeholder if class is missing
            return df
        else:
             st.warning(f"Sample data file not found at {data_path}")
             return None
    except Exception as e:
        st.error(f"Error loading sample data: {e}")
        logger.error(f"Sample data loading failed: {e}\n{traceback.format_exc()}")
        return None


# --- UI Component Functions ---

def create_feature_input_form():
    """Create interactive feature input form using sliders."""
    st.subheader("✨ Object Properties")
    st.markdown("Adjust the sliders to match the observed properties:")
    form_col1, form_col2 = st.columns(2)

    with form_col1:
        st.markdown("##### Magnitude Measurements")
        # Tooltips added via help parameter
        u_mag = st.slider("U Magnitude (UV)", 10.0, 30.0, 18.8, 0.1, help="Brightness in Ultraviolet filter (lower = brighter)")
        g_mag = st.slider("G Magnitude (Green)", 10.0, 30.0, 17.5, 0.1, help="Brightness in Green filter (lower = brighter)")
        r_mag = st.slider("R Magnitude (Red)", 10.0, 30.0, 16.9, 0.1, help="Brightness in Red filter (lower = brighter)")
        i_mag = st.slider("I Magnitude (Near IR)", 10.0, 30.0, 16.5, 0.1, help="Brightness in Near Infrared filter (lower = brighter)")
        z_mag = st.slider("Z Magnitude (Infrared)", 10.0, 30.0, 16.3, 0.1, help="Brightness in Infrared filter (lower = brighter)")

    with form_col2:
        st.markdown("##### Physical Properties")
        redshift = st.slider("Redshift (z)", -0.1, 7.5, 0.5, 0.01, help="Measure of how much light is stretched (related to distance/velocity)")

        st.markdown("---") # Visual separator
        with st.expander("ℹ️ Feature Explanations"):
            st.info("""
            * **Magnitudes (u, g, r, i, z):** Measure the brightness of the object in different color filters used by the SDSS telescope. **Lower** magnitude values indicate **brighter** objects.
            * **Redshift (z):** Indicates how much the object's light has been stretched due to the expansion of the universe. Higher redshift generally means the object is farther away and moving away faster.
            """)

    # Return dictionary with the 6 raw features
    return {
        'u': u_mag, 'g': g_mag, 'r': r_mag, 'i': i_mag, 'z': z_mag,
        'redshift': redshift
    }

def display_prediction_results(prediction_idx, probabilities, class_names):
    """Display prediction results with updated styling."""
    st.subheader("🎯 Prediction Result")

    predicted_class_name = class_names[prediction_idx]
    confidence = probabilities[prediction_idx] * 100

    # Display using st.metric for a cleaner look within the card
    pred_col1, pred_col2 = st.columns(2)
    with pred_col1:
        st.metric(label="Predicted Class", value=predicted_class_name)
    with pred_col2:
        st.metric(label="Confidence", value=f"{confidence:.2f}%")


    st.markdown("##### Class Probabilities")
    prob_df = pd.DataFrame({
        'Class': class_names,
        'Probability': probabilities
    }).sort_values('Probability', ascending=False)

    fig = px.bar(prob_df, x='Probability', y='Class', orientation='h',
                 color='Probability', color_continuous_scale=px.colors.sequential.Viridis, # Consistent palette
                 text='Probability',
                 # title="Confidence Score per Class" # Removed title, implied by subheader
                 )
    fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig.update_layout(height=250, showlegend=False,
                      yaxis={'categoryorder':'total ascending'},
                      margin=dict(l=10, r=10, t=20, b=10), # Reduce margins
                      xaxis_range=[0,1]) # Ensure x-axis goes from 0 to 1
    st.plotly_chart(fig, use_container_width=True)

def create_shap_analysis(pipeline: PredictionPipeline, X_sample: pd.DataFrame):
    """
    Create SHAP analysis visualization for a single sample.
    Uses extracted scaler and classifier. Includes robust value extraction.
    """
    st.subheader("🧐 Local Model Interpretability (SHAP)")
    st.markdown("Shows how much each feature value contributed to this specific prediction compared to the average prediction.")

    # --- Input Validation ---
    if not pipeline.is_loaded:
        st.error("Prediction pipeline is not loaded.")
        return
    classifier = pipeline.classifier
    scaler = pipeline.scaler_step # Can be None if not found/used
    if classifier is None:
         st.error("Classifier could not be extracted from the pipeline. Cannot run SHAP.")
         return

    try:
        # --- Scale the Input Sample ---
        # X_sample has 13 engineered features (from test.csv)
        if scaler:
             X_sample_scaled = scaler.transform(X_sample)
             feature_names = pipeline.feature_names # Use the known 13 feature names
             X_sample_scaled_df = pd.DataFrame(X_sample_scaled, columns=feature_names, index=X_sample.index)
             logger.info("Sample scaled for SHAP analysis.")
        else:
             X_sample_scaled_df = X_sample[pipeline.feature_names]
             st.warning("Scaler step not found in loaded model. Proceeding with SHAP on potentially unscaled data. Interpretability might be affected.")
             logger.warning("Proceeding with SHAP on potentially unscaled data.")

        # --- Initialize SHAP Explainer ---
        model_name = type(classifier).__name__
        logger.info(f"Initializing SHAP explainer for {model_name}...")
        with st.spinner(f"Calculating SHAP values using {model_name} explainer..."): # Show spinner
            if model_name in ['RandomForestClassifier', 'XGBClassifier', 'GradientBoostingClassifier']:
                explainer = shap.TreeExplainer(classifier)
                expected_value = explainer.expected_value
                shap_values = explainer.shap_values(X_sample_scaled_df) # Explain the single scaled sample
                logger.info("TreeExplainer SHAP values calculated.")

            else: # Handle non-tree models
                st.info(f"Using KernelExplainer for {model_name}. This can take several minutes...")

                background_data_full = load_sample_data(pipeline) # Pass pipeline to load_sample_data
                if background_data_full is None:
                     st.error("Cannot run KernelExplainer: Failed to load background data (test.csv).")
                     return

                background_data = background_data_full.drop(columns=['class', 'class_name'], errors='ignore')
                background_data = background_data[pipeline.feature_names] # Ensure correct features

                if scaler:
                     scaled_background = scaler.transform(background_data)
                else:
                     scaled_background = background_data

                # Use shap.sample for a representative background subset
                background_sample = shap.sample(scaled_background, 50, random_state=42)
                logger.info(f"Created background sample ({background_sample.shape[0]} rows) for KernelExplainer.")

                # Wrapper function for predict_proba
                def predict_proba_func(data_np):
                    data_df_func = pd.DataFrame(data_np, columns=pipeline.feature_names)
                    return classifier.predict_proba(data_df_func)

                explainer = shap.KernelExplainer(predict_proba_func, background_sample)
                expected_value = explainer.expected_value
                shap_values = explainer.shap_values(X_sample_scaled_df) # This is the slow part
                logger.info("KernelExplainer SHAP values calculated.")

        # --- Plot Waterfall for the Predicted Class ---
        st.markdown("##### Feature Contributions (Waterfall Plot)")

        prediction_idx = classifier.predict(X_sample_scaled_df)[0]
        predicted_class_name = pipeline.label_encoder.classes_[prediction_idx]

        st.markdown(f"**Showing contributions towards predicted class: `{predicted_class_name}`**")

        # --- Robust SHAP Value Extraction for Plotting ---
        shap_values_one_class = None
        expected_value_one_class = None

        if isinstance(shap_values, list) and len(shap_values) == len(pipeline.label_encoder.classes_):
             # Standard multi-class: list of arrays [ (n_samples, n_features), ... ]
             if shap_values[prediction_idx].shape[0] == 1:
                 shap_values_one_class = shap_values[prediction_idx][0, :] # Shape (n_features,)
             else: raise ValueError("Unexpected SHAP array shape.")

             if isinstance(expected_value, (list, np.ndarray)) and len(expected_value) == len(pipeline.label_encoder.classes_):
                 expected_value_one_class = expected_value[prediction_idx]
             else: expected_value_one_class = expected_value[0] if isinstance(expected_value, (list, np.ndarray)) else 0

        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
             # Alt multi-class (rare for 1 sample?): array (n_samples, n_features, n_classes)
              if shap_values.shape[0] == 1 and shap_values.shape[2] == len(pipeline.label_encoder.classes_):
                  shap_values_one_class = shap_values[0, :, prediction_idx] # Shape (n_features,)
                  if isinstance(expected_value, (list, np.ndarray)): expected_value_one_class = expected_value[prediction_idx]
                  else: expected_value_one_class = expected_value[0] if isinstance(expected_value, (list, np.ndarray)) else 0
              else: raise ValueError("Unexpected SHAP array shape.")
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 2 and shap_values.shape[0] == 1:
             # Binary classification or regression output for 1 sample: array (1, n_features)
             shap_values_one_class = shap_values[0, :] # Shape (n_features,)
             expected_value_one_class = expected_value # Should be single value
        else:
            raise TypeError(f"Unhandled SHAP value format. Type: {type(shap_values)}")

        if shap_values_one_class is None or expected_value_one_class is None:
             st.error("Failed to extract SHAP values for plotting.")
             return

        # Create the SHAP Explanation object
        shap_explanation = shap.Explanation(
            values=shap_values_one_class,
            base_values=expected_value_one_class,
            data=X_sample_scaled_df.iloc[0].values, # Use scaled data
            feature_names=pipeline.feature_names
        )

        # Plot using Matplotlib - Use Streamlit's theme context for better integration
        plt.style.use('seaborn-v0_8-whitegrid') # Cleaner Matplotlib style
        fig, ax = plt.subplots()
        shap.plots.waterfall(shap_explanation, max_display=13, show=False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig) # Close figure

    except Exception as e:
        st.error(f"❌ Error creating SHAP analysis: {str(e)}")
        logger.error(f"SHAP analysis failed: {e}\n{traceback.format_exc()}")


def create_model_performance_dashboard():
    """Create model performance dashboard from saved comparison CSV with improved aesthetics."""
    st.subheader("📈 Model Performance Comparison")
    st.markdown("Comparing metrics across different models trained during the pipeline run:")

    try:
        comparison_path = Path("reports/model_performance_comparison.csv")
        if not comparison_path.exists():
            st.warning(f"Performance comparison file not found: `{comparison_path}`.")
            st.info("Please run the training pipeline (`run_pipeline.py`) to generate this file.")
            return

        df = pd.read_csv(comparison_path)

        # --- Robustly find the Model column ---
        model_col_name = None
        potential_model_cols = ['Model', 'model', 'Algorithm'] # Common names
        # Check standard names first
        for col in potential_model_cols:
             if col in df.columns:
                 model_col_name = col
                 break
        # If not found, check if first column acts as index
        if model_col_name is None:
             first_col = df.columns[0]
             # Heuristic: First col might be index if unique & not just numbers or 'Unnamed'
             if df[first_col].nunique() == len(df) and not pd.api.types.is_numeric_dtype(df[first_col]):
                 model_col_name = first_col
                 logger.info(f"Using first column '{model_col_name}' as model names.")
             else: # Fallback if no clear model name column
                 st.warning(f"Could not reliably identify model name column. Using index.")
                 df['Model_Index'] = df.index.astype(str) # Create temp index column
                 model_col_name = 'Model_Index'


        model_names = df[model_col_name].tolist()

        # --- Setup Plot ---
        metrics_setup = [
            {'title': 'Accuracy', 'cols': ['Train_Accuracy', 'Val_Accuracy', 'Test_Accuracy'], 'range': [0.8, 1.01]}, # Set specific ranges
            {'title': 'F1 Score (Macro)', 'cols': ['Val_F1_Macro', 'Test_F1_Macro'], 'range': [0.7, 1.01]},
            {'title': 'ROC AUC (Macro OVR)', 'cols': ['Val_ROC_AUC', 'Test_ROC_AUC'], 'range': [0.9, 1.01]}
        ]
        # Updated Color Palette (consider using Streamlit theme colors potentially)
        # Using Plotly's default qualitative 'Pastel' palette might be cleaner
        colors = {'Train': px.colors.qualitative.Pastel[0],
                  'Validation': px.colors.qualitative.Pastel[1],
                  'Test': px.colors.qualitative.Pastel[2]}

        fig = make_subplots(
            rows=len(metrics_setup), cols=1,
            subplot_titles=[m['title'] for m in metrics_setup],
            shared_xaxes=True,
            vertical_spacing=0.15
        )

        # --- Add Traces ---
        for row_num, metric_info in enumerate(metrics_setup, start=1):
            show_legend_row = (row_num == 1)
            available_cols = [col for col in metric_info['cols'] if col in df.columns]

            for col_name in available_cols:
                 bar_name = "Unknown"
                 bar_color = "grey"

                 if 'Train_' in col_name: bar_name, bar_color = 'Train', colors['Train']
                 elif 'Val_' in col_name: bar_name, bar_color = 'Validation', colors['Validation']
                 elif 'Test_' in col_name: bar_name, bar_color = 'Test', colors['Test']

                 fig.add_trace(go.Bar(
                     x=model_names,
                     y=df[col_name],
                     name=bar_name,
                     marker_color=bar_color,
                     legendgroup='splits', # Group legends
                     showlegend=show_legend_row,
                     text=df[col_name].apply(lambda x: f'{x:.3f}'),
                     textposition='outside',
                     hoverinfo='x+y+name',
                     hovertemplate=f"<b>Model:</b> %{{x}}<br><b>{bar_name} {metric_info['title']}:</b> %{{y:.4f}}<extra></extra>"
                     ), row=row_num, col=1
                 )
            # Update y-axis range and title for the row
            fig.update_yaxes(title_text=metric_info['title'], range=metric_info['range'], row=row_num, col=1)

        # --- Final Layout Updates ---
        fig.update_layout(
            height=250 * len(metrics_setup) + 100, # Adjust height based on plots + margin
            barmode='group',
            bargap=0.15, # Space between bars of different models
            bargroupgap=0.1, # Space between bars of same model
            legend_title_text='Dataset Split',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified",
            margin=dict(l=60, r=20, t=60, b=100), # Ensure enough bottom margin for labels
            # title_text="Model Performance Comparison", title_x=0.5, # Title set by subplots
            font=dict(family="sans-serif"), # Use default sans-serif
            template="plotly_white" # Cleaner template
        )
        fig.update_xaxes(tickangle=-45, row=len(metrics_setup), col=1) # Angle bottom labels

        st.plotly_chart(fig, use_container_width=True)

    except FileNotFoundError:
         st.warning(f"Performance comparison file not found.") # Handled above, but good practice
    except KeyError as e:
         st.error(f"❌ Error plotting performance data: A required column is missing from the CSV: {e}")
         st.info("Please check 'reports/model_performance_comparison.csv' and ensure the training pipeline saves all necessary metric columns (e.g., Train_Accuracy, Val_F1_Macro).")
         logger.error(f"Performance dashboard KeyError: {e}\n{traceback.format_exc()}")
    except Exception as e:
        st.error(f"❌ Error loading or plotting performance data: {e}")
        logger.error(f"Performance dashboard error: {e}\n{traceback.format_exc()}")


def create_data_exploration():
    """Create data exploration section using sample test data."""
    st.subheader("📊 Data Exploration")
    st.markdown("Exploring the distribution of features in the processed **test set**.")

    # Need pipeline loaded to map class names
    if pipeline is None:
        st.error("Pipeline not loaded. Cannot proceed with data exploration.")
        return

    sample_data = load_sample_data(pipeline) # Pass pipeline to map names

    if sample_data is not None:
        st.markdown("##### Test Set Overview")
        exp_col1, exp_col2, exp_col3, exp_col4 = st.columns(4)
        exp_col1.metric("Samples", len(sample_data))
        exp_col2.metric("Features", len(pipeline.feature_names) if pipeline.feature_names else "N/A")
        # Use mapped class names if available
        class_col = 'class_name' if 'class_name' in sample_data else 'class'
        exp_col3.metric("Classes", sample_data[class_col].nunique())
        exp_col4.metric("Missing Values", sample_data.isnull().sum().sum())

        st.markdown("##### Sample Data Preview")
        st.dataframe(sample_data.head())
        st.caption("Showing first 5 rows of the processed test data.")

        st.markdown("##### Class Distribution")
        if 'class_name' in sample_data:
             class_counts = sample_data['class_name'].value_counts()
             fig_pie = px.pie(class_counts, values=class_counts.values, names=class_counts.index,
                         title="Distribution of Actual Classes in Test Set", hole=0.4,
                         color_discrete_sequence=px.colors.qualitative.Pastel) # Consistent colors
             fig_pie.update_layout(height=350, legend_title_text='Class')
             st.plotly_chart(fig_pie, use_container_width=True)
        else:
             st.warning("Could not display class distribution by name.")

        st.divider() # Visual separator

        st.markdown("##### Feature Distributions by Class")
        # Use known feature names, exclude target/mapped target
        numeric_features = list(pipeline.feature_names) if pipeline.feature_names else []

        if not numeric_features:
             st.warning("Feature names not loaded. Cannot display distributions.")
             return

        selected_feature = st.selectbox(
             "Select a feature to visualize:",
             numeric_features,
             index=numeric_features.index('redshift') if 'redshift' in numeric_features else 0,
             key="data_exp_feature_select" # Unique key for selectbox
        )

        if selected_feature:
            # Use Plotly histogram for interactive exploration
            fig_hist = px.histogram(sample_data, x=selected_feature,
                                    color='class_name' if 'class_name' in sample_data else None,
                                    marginal="box", # Show distribution summary
                                    title=f"Distribution of '{selected_feature}' by Class",
                                    opacity=0.75,
                                    barmode='overlay', # Overlay histograms
                                    color_discrete_sequence=px.colors.qualitative.Pastel, # Consistent colors
                                    template="plotly_white") # Cleaner template
            fig_hist.update_layout(height=450, legend_title_text='Actual Class')
            st.plotly_chart(fig_hist, use_container_width=True)

    else:
        st.error("Sample data (test.csv) could not be loaded. Cannot display data exploration.")


# --- Main Application Logic ---
def main():
    """Main Streamlit application function."""

    st.markdown('<h1 class="main-header">🔭 Stellar Classify Pro</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Classify Stars, Galaxies, and Quasars using Machine Learning ✨</p>', unsafe_allow_html=True)

    # --- Load Model Pipeline ---
    global pipeline # Declare pipeline as global for access in helper functions
    pipeline = load_model_artifacts()

    if pipeline is None:
        # Error message is shown in load_model_artifacts
        st.stop() # Halt execution if model loading fails

    # --- Create Tabs ---
    tab_titles = ["🎯 Predict", "📊 Explore Data", "📈 Performance", "🧐 Interpretability", "ℹ️ About"]
    tab1, tab2, tab3, tab4, tab5 = st.tabs(tab_titles)

    # --- Tab 1: Predict ---
    with tab1:
        with st.container(): # Use container for styling
            st.header("Make a Real-time Prediction")
            st.write("Adjust the sliders below to match the observed properties of a celestial object.")
            st.divider()
            features_dict = create_feature_input_form()
            st.divider()

            if st.button("🚀 Classify Object", type="primary", use_container_width=True):
                with st.spinner("Classifying... Please wait."): # Add spinner for feedback
                    try:
                        input_df = pd.DataFrame([features_dict])
                        logger.info(f"Input data for prediction:\n{input_df}")

                        prediction_idx, probabilities = pipeline.predict(input_df)

                        display_prediction_results(
                            prediction_idx[0],
                            probabilities[0],
                            pipeline.label_encoder.classes_
                        )

                    except Exception as e:
                        st.error(f"❌ Prediction failed: {str(e)}")
                        logger.error(f"Prediction error: {e}\n{traceback.format_exc()}")

    # --- Tab 2: Explore Data ---
    with tab2:
         with st.container():
             create_data_exploration()

    # --- Tab 3: Performance ---
    with tab3:
        with st.container():
            create_model_performance_dashboard()

    # --- Tab 4: Interpretability ---
    with tab4:
        with st.container():
            st.header("Understand the Model's Decision")
            st.write("Select a sample from the test set to see which features influenced its classification the most.")
            st.divider()

            sample_data = load_sample_data(pipeline) # Pass pipeline to map names

            if sample_data is not None:
                # Limit samples in dropdown for performance
                num_samples_to_show = min(20, len(sample_data))

                # Create display options with actual class names
                display_options = [f"Sample {i} (Actual: {sample_data.loc[i, 'class_name']})"
                                     if 'class_name' in sample_data else f"Sample {i}"
                                     for i in range(num_samples_to_show)]

                selected_option = st.selectbox(
                    "Select a sample to explain:",
                    display_options,
                    index=0,
                    key="shap_sample_select" # Unique key
                )

                # Extract the index from the selected option string
                try:
                    sample_idx = int(selected_option.split(" ")[1])
                except (IndexError, ValueError):
                    st.error("Could not parse sample index from selection.")
                    st.stop()


                # Get the specific sample row (needs the 13 engineered features)
                cols_to_drop = ['class', 'class_name'] # Drop target/mapped name
                X_sample = sample_data.iloc[[sample_idx]].drop(columns=cols_to_drop, errors='ignore')

                # Ensure it has the correct 13 columns in the right order
                if pipeline.feature_names:
                     try:
                         X_sample = X_sample[pipeline.feature_names]
                     except KeyError as e:
                          st.error(f"Selected sample is missing expected features: {e}")
                          st.stop()
                else:
                    st.error("Feature names not loaded from pipeline. Cannot select sample features correctly.")
                    st.stop()

                st.markdown(f"**Analyzing Features for Sample `{sample_idx}`:**")
                st.dataframe(X_sample)
                st.divider()

                # Generate and display SHAP analysis for this sample
                create_shap_analysis(pipeline, X_sample)

            else:
                st.error("Sample data (test.csv) needed for SHAP analysis could not be loaded.")

    # --- Tab 5: About ---
    with tab5:
        with st.container():
            st.header("About This Application")
            # Consider adding a real, relevant image later
            # st.image("path/to/your/banner.jpg", caption="Image Credit: SDSS or similar")
            st.markdown("""
            #### What is This?
            This **Stellar Classify Pro** application uses machine learning to categorize celestial objects (Stars, Galaxies, Quasars) based on observational data, primarily from surveys like the Sloan Digital Sky Survey (SDSS).

            #### How it Works
            1.  **Data:** Trained on processed SDSS data containing photometric measurements and redshift.
            2.  **Features:** Utilizes brightness in 5 filters (u, g, r, i, z), redshift, plus derived features like color indices (e.g., u-g) and magnitude statistics.
            3.  **Model:** Employs a **Random Forest Classifier** (selected as the best performer) embedded within a preprocessing pipeline that includes data scaling (`StandardScaler`) and handling class imbalance (`SMOTE`).
            4.  **Prediction:** Accepts feature values via sliders, engineers necessary features, processes them through the pipeline, and outputs the predicted class with confidence probabilities.
            5.  **Interpretability:** Leverages **SHAP** (SHapley Additive exPlanations) to provide insights into *why* the model classified a specific object the way it did, showing the impact of each feature on that single prediction.

            ---
            #### Technical Details
            * **Core Libraries:** Python, Scikit-learn, Imbalanced-learn, Pandas, NumPy
            * **Interpretability:** SHAP
            * **Web Framework:** Streamlit
            * **Visualization:** Plotly, Matplotlib
            * **Workflow:** Built using a modular pipeline structure (data ingestion, transformation, training, prediction).

            ---
            #### Project Context
            This application serves as a practical demonstration of an end-to-end machine learning project for astronomical data analysis, emphasizing model performance, deployment, and explainability.
            """)

    # --- Footer ---
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888; margin-top: 2rem; font-size: 0.9rem;">
        <p>Stellar Classify Pro | Capstone Project | Developed by AbdulRahman Sami | Data Source: SDSS DR14</p>
    </div>
    """, unsafe_allow_html=True)

# --- Global variable for the loaded pipeline ---
pipeline: PredictionPipeline = None

# --- Run the main function ---
if __name__ == "__main__":
    main()