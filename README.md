# Healthcare Insurance Cost Prediction System

A comprehensive machine learning system for predicting healthcare insurance costs, risk scores, and high-risk patient classification. The system includes three trained models and an interactive Streamlit dashboard for predictions and analysis.

## Features

- **Three ML Models:**
  - Annual Medical Cost Prediction (Regression)
  - Risk Score Prediction (Regression)
  - High-Risk Patient Classification (Binary Classification)

- **Interactive Dashboard:**
  - Real-time predictions with user input
  - Data visualizations and analysis
  - Model performance metrics
  - Feature importance analysis
  - Data explorer with filtering

- **Comprehensive Analysis:**
  - Cost distribution analysis
  - Risk segmentation
  - Demographic insights
  - Health metrics correlation

## Dataset

The dataset contains 100,000 records with 54 features including:
- Demographics (age, sex, region, income, education, etc.)
- Health metrics (BMI, blood pressure, lab values, etc.)
- Insurance plan details (plan type, tier, deductible, etc.)
- Chronic conditions (hypertension, diabetes, etc.)
- Medical procedures and claims history

**Target Variables:**
- `annual_medical_cost`: Annual medical expenses (regression target)
- `risk_score`: Risk score from 0 to 1 (regression target)
- `is_high_risk`: Binary classification (0 = Low Risk, 1 = High Risk)

## Installation

1. **Clone or download this repository**

2. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Data Exploration

Explore the dataset structure and statistics:
```bash
python explore_data.py
```

### 2. Train Models

Train all three models (this will create the preprocessing pipeline and save trained models):
```bash
python train_models.py
```

This will:
- Create and save the preprocessing pipeline
- Train models for cost, risk score, and high-risk classification
- Evaluate models and select the best performing ones
- Save all models as `.pkl` files in the `models/` directory

**Expected Training Time:** ~5-10 minutes (depending on hardware)

### 3. Run Streamlit Dashboard

Launch the interactive dashboard:
```bash
streamlit run app.py
```

The dashboard will open in your default web browser at `http://localhost:8501`

## Project Structure

```
hybrid-actuarial-dss/
├── medical_insurance.csv          # Dataset (100k rows)
├── app.py                         # Streamlit dashboard
├── explore_data.py                # Data exploration script
├── preprocessing.py               # Preprocessing pipeline
├── train_models.py                # Model training script
├── models/                        # Saved models directory
│   ├── preprocessor.pkl          # Preprocessing pipeline
│   ├── preprocessor_metadata.pkl # Feature metadata
│   ├── cost_model.pkl            # Cost prediction model
│   ├── risk_score_model.pkl      # Risk score model
│   ├── high_risk_model.pkl       # High-risk classification model
│   └── model_metrics.pkl         # Model performance metrics
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── .gitignore                     # Git ignore file
```

## Model Performance

### Cost Regression Model
- **Best Model:** Gradient Boosting
- **Test RMSE:** ~130.07
- **Test R²:** ~0.9983

### Risk Score Regression Model
- **Best Model:** Random Forest
- **Test RMSE:** ~0.0030
- **Test R²:** ~0.9999

### High-Risk Classification Model
- **Best Model:** XGBoost
- **Test Accuracy:** ~100%
- **Test AUC:** ~1.0000

## Dashboard Features

### 1. Predictions Page
- Input form for all patient features
- Real-time predictions for all three targets
- Risk interpretation and confidence scores

### 2. Data Visualizations
- Cost distribution histograms
- Risk score analysis
- Demographic breakdowns
- Health metrics correlations

### 3. Model Performance
- Detailed metrics for each model
- Feature importance visualizations
- Training vs test performance comparison

### 4. Data Explorer
- Interactive data table
- Filtering capabilities
- Statistical summaries

### 5. Insights
- Key factors affecting costs
- Risk factor analysis
- Feature importance comparison across models

## Technical Details

### Preprocessing
- Missing value imputation (median for numerical, mode for categorical)
- Standard scaling for numerical features
- One-hot encoding for categorical features
- Handles unknown categories in new data

### Models Used
- **Random Forest:** Ensemble method with multiple decision trees
- **XGBoost:** Gradient boosting framework
- **Gradient Boosting:** Sequential ensemble learning
- **Linear/Logistic Regression:** Baseline models

### Evaluation Metrics
- **Regression:** MAE, RMSE, R²
- **Classification:** Accuracy, Precision, Recall, F1, ROC-AUC

## Requirements

- Python 3.8+
- pandas >= 2.0.0
- numpy >= 1.24.0
- scikit-learn >= 1.3.0
- xgboost >= 2.0.0
- streamlit >= 1.28.0
- plotly >= 5.17.0
- joblib >= 1.3.0

## Notes

- The dataset contains some missing values in `alcohol_freq` (30% missing), which are handled by the preprocessing pipeline
- Models are saved in `.pkl` format using joblib for efficient serialization
- The dashboard caches data and models for better performance
- All models are trained on 80% of the data, tested on 20%

## License

This project is for educational and research purposes.

## Author

Healthcare Insurance Prediction System - ML Project


