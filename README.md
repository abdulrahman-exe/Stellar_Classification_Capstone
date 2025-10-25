# 🔭 Stellar Classify Pro: End-to-End Stellar Object Classification 🌟

[![Python Version](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE) 
[![Open in Streamlit](https://img.shields.io/badge/Open%20in%20Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://abdulrahmansami-exe-stellar-classification-capstone.streamlit.app)

**Classify the cosmos!** This project provides a complete machine learning pipeline to categorize celestial objects (Stars ⭐, Galaxies 🌌, Quasars ✨) using data from the Sloan Digital Sky Survey (SDSS). It covers the entire workflow: data ingestion, cleaning, feature engineering, model training & comparison, deployment via an interactive Streamlit app, and in-depth model interpretability using SHAP.

---

### 🌐 Live Demo

**[🔭 Try the Streamlit App](https://abdulrahmansami-exe-stellar-classification-capstone.streamlit.app)**

![Streamlit App Demo](assets/streamlit_demo.gif)

---

## 🎯 Project Goal

To build a robust, reproducible, and interpretable end-to-end pipeline for classifying stellar objects, deployable as an easy-to-use web application.

---
## 📓 Notebooks

- **[EDA Notebook](notebooks/EDA.ipynb)** - Exploratory Data Analysis
- **[Complete Pipeline Notebook](https://nbviewer.org/github/abdulrahman-exe/Stellar_Classification_Capstone/blob/main/notebooks/Stellar_Classification_Capstone_Complete.ipynb)** - Full end-to-end workflow *(View on nbviewer)*


---
## 🚀 Key Features

* **🌌 Data Processing:** Automated pipeline for loading, validating, cleaning (handling `-9999` values), engineering new features (color indices, magnitude stats), and splitting SDSS data while preventing data leakage.
* **⚖️ Class Imbalance Handling:** Utilizes SMOTE within the pipeline to address the uneven distribution of object classes.
* **🧠 Model Training & Selection:** Trains and rigorously evaluates multiple classifiers (Random Forest, XGBoost, Gradient Boosting, SVM, Logistic Regression) using metrics like Accuracy, F1-Score, and ROC AUC. Automatically selects and saves the best-performing model.
* **💡 Interpretability:** Employs **SHAP** (SHapley Additive exPlanations) for both global feature importance and local (per-prediction) explanations via waterfall plots.
* **🖥️ Interactive Web App:** A Streamlit application offering:
  * Real-time classification using intuitive sliders.
  * Exploration of the processed test dataset with visualizations.
  * Dashboard comparing the performance of trained models.
  * Interactive SHAP explanations for selected test samples.
* **✅ Reproducibility:** Includes `requirements.txt` and clear setup instructions.

---

## 🛠️ Tech Stack

* **Core:** Python 3.11+, Pandas, NumPy
* **ML Pipeline:** Scikit-learn, Imbalanced-learn, XGBoost
* **Interpretability:** SHAP
* **Web App & Viz:** Streamlit, Plotly, Matplotlib, Seaborn
* **Persistence:** Joblib

---

## ⚙️ Setup & Usage

1. **Clone the Repository:**

   ```bash
   git clone [https://github.com/abdulrahmansami_exe/Stellar_Classification_Capstone.git](https://github.com/abdulrahmansami_exe/Stellar_Classification_Capstone.git)
   cd Stellar_Classification_Capstone
   ```
2. **Set Up Environment:**

   ```bash
   # Create & activate virtual environment (adjust for your OS)
   python -m venv .venv
   source .venv/bin/activate # Or .\.venv\Scripts\activate on Windows

   # Install dependencies
   pip install -r requirements.txt
   ```

   *(**Note:** The raw data `star_classification.csv` and saved model `best_model.pkl` are included in the repository for immediate use).*
3. **Run the Streamlit App:**

   ```bash
   streamlit run app.py
   ```

   Navigate to the provided URL (e.g., `http://localhost:8501`) in your browser.
4. **(Optional) Re-run Training:**

   * To retrain all models and save the best one based on your data/environment:

   ```bash
   python run_pipeline.py
   ```

---

## 📂 Project Structure

```
Stellar_Classification_Capstone/
├── src/
│   ├── components/
│   │   ├── data_ingestion.py      # Data loading and validation
│   │   ├── data_transformation.py # Data cleaning and feature engineering
│   │   └── model_trainer.py       # Model training and evaluation
│   ├── pipelines/
│   │   ├── train_pipeline.py      # Complete training workflow
│   │   └── predict_pipeline.py    # Prediction pipeline
│   ├── config.py                  # Configuration and constants
│   ├── utils.py                   # Utility functions (plotting, logging, SHAP)
│   └── check_data.py              # Data quality validation (if applicable)
├── data/
│   ├── raw/                       # Raw SDSS data (e.g., star_classification.csv)
│   └── processed/                 # Processed datasets (train, val, test splits)
├── saved_models/                  # Trained models and preprocessors (.pkl files)
├── reports/
│   ├── figures/                   # Saved plots and visualizations
│   ├── final_model_summary.json   # A JSON file with the final model's summary
│   └── model_performance_comparison.csv # Model comparison results
├── notebooks/                     # Jupyter notebooks for EDA and experimentation
├── assets/                        # Static assets like images/GIFs
├── docs/                          # Project documentation and reports
│   ├── Description.docx           # Project description document
│   ├── Stellar_Classification_Report.pptx  # Your PPT presentation file
├── .gitignore                     # Specifies files/folders Git should ignore
├── app.py                         # Streamlit web application script
├── LICENSE                        # Project license file (e.g., MIT)
├── README.md                      # Project overview and instructions
├── requirements.txt               # Python package dependencies
└── run_pipeline.py                # Script to execute the full training pipeline
---
```
## 📊 Results

The **Random Forest Classifier** (within the Scaler -> SMOTE -> Classifier pipeline) emerged as the top performer, achieving:

* **Test Accuracy:** ~97.7%
* **Test F1 (Macro):** ~0.973

Key features driving predictions include `redshift`, color indices (like `u_g`, `g_r`), and magnitude standard deviation (`mag_std`), as confirmed by SHAP analysis.

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE] file for details.

---

**Happy Classifying! 🌌**
```
