# Hybrid Actuarial Decision Support System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-Educational-green.svg)
![ML](https://img.shields.io/badge/ML-Machine%20Learning-orange.svg)

**An intelligent machine learning system for healthcare insurance cost prediction, risk assessment, and actuarial analysis**

[Features](#-features) • [Quick Start](#-quick-start) • [Model Performance](#-model-performance) • [Dashboard](#-dashboard)

</div>

---

## 📋 Overview

The Hybrid Actuarial Decision Support System (DSS) is a comprehensive machine learning platform designed for insurance companies to accurately predict medical costs, assess health risks, and calculate premium recommendations. The system leverages state-of-the-art ensemble models to provide actionable insights for underwriting and actuarial decision-making.

### Key Capabilities

- **💰 Cost Prediction**: Forecast annual medical expenses with high accuracy (R² > 0.99)
- **⚠️ Risk Assessment**: Classify patients into risk categories with automated underwriting flags
- **💳 Premium Calculation**: Generate risk-adjusted premium recommendations with actuarial load factors
- **📊 Interactive Dashboard**: Streamlit-based web interface for real-time analysis and reporting

---

## ✨ Features

### Machine Learning Models

| Model | Algorithm | Target | Performance |
|-------|-----------|--------|-------------|
| **Cost Prediction** | Gradient Boosting | Annual Medical Cost | R² = 0.9983, RMSE = 130.07 |
| **Risk Scoring** | Random Forest | Risk Score (0-1) | R² = 0.9999, RMSE = 0.0030 |
| **Risk Classification** | XGBoost | High-Risk Binary | Accuracy = 100%, AUC = 1.0000 |

### Dashboard Capabilities

- ✅ **Real-time Predictions** - Instant results with comprehensive input forms
- ✅ **Actuarial Analysis** - Risk load calculation and premium recommendations
- ✅ **What-If Simulator** - Scenario analysis with variable adjustments
- ✅ **Report Generation** - Export results to CSV and PDF formats
- ✅ **SHAP Explanations** - Feature importance visualization
- ✅ **Risk Flagging** - Automated identification of underwriting concerns

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd hybrid-actuarial-dss
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the models** (First time only)
   ```bash
   python train_models.py
   ```
   *Expected time: 5-10 minutes*

4. **Launch the dashboard**
   ```bash
   streamlit run app.py
   ```

The dashboard will automatically open at `http://localhost:8501`

---

## 📁 Project Structure

```
hybrid-actuarial-dss/
│
├── 📄 app.py                      # Main Streamlit dashboard application
├── 📊 medical_insurance.csv       # Dataset (100,000 records, 54 features)
├── 🔍 explore_data.py             # Data exploration utilities
├── ⚙️  preprocessing.py            # Preprocessing pipeline
├── 🎯 train_models.py             # Model training script
├── 📦 requirements.txt            # Python dependencies
│
└── 📂 models/                     # Trained models directory
    ├── preprocessor.pkl          # Data preprocessing pipeline
    ├── preprocessor_metadata.pkl # Feature metadata
    ├── cost_model.pkl            # Medical cost prediction model
    ├── risk_score_model.pkl      # Risk score regression model
    ├── high_risk_model.pkl       # High-risk classification model
    └── model_metrics.pkl         # Performance metrics
```

---

## 🎯 Model Performance

### Cost Regression Model
```
Algorithm: Gradient Boosting Regressor
├─ Test RMSE: 130.07
├─ Test R²:   0.9983
└─ Accuracy:  Excellent
```

### Risk Score Model
```
Algorithm: Random Forest Regressor
├─ Test RMSE: 0.0030
├─ Test R²:   0.9999
└─ Accuracy:  Excellent
```

### High-Risk Classification
```
Algorithm: XGBoost Classifier
├─ Accuracy:  100%
├─ AUC-ROC:   1.0000
└─ Precision: Excellent
```

---

## 🖥️ Dashboard

The interactive Streamlit dashboard provides a comprehensive interface for actuarial analysis:

### Tab Overview

| Tab | Description |
|-----|-------------|
| **📖 Manual** | Comprehensive user guide and metric explanations |
| **📝 Input** | Patient information input form with validation |
| **📊 Results** | Detailed predictions, risk flags, and premium calculations |
| **🔬 Simulator** | What-if analysis with variable adjustments |
| **📄 Report** | Export underwriting summaries to CSV/PDF |

### Input Categories

- **Demographics**: Age, gender, region, income, education, marital status
- **Health Metrics**: BMI, blood pressure, smoking status, alcohol frequency
- **Chronic Conditions**: Hypertension, diabetes, cardiovascular disease, etc.
- **Insurance Details**: Plan type, network tier, deductible, copay, premium

### Output Features

- **Cost Prediction**: Annual medical expense forecast
- **Risk Score**: Numeric risk assessment (0-1 scale)
- **Risk Classification**: High/Low risk categorization
- **Premium Recommendation**: Risk-adjusted premium with load factors
- **Underwriting Flags**: Automated risk factor identification
- **Feature Importance**: SHAP values for model interpretability

---

## 🔧 Technical Details

### Dataset

- **Size**: 100,000 records
- **Features**: 54 variables (demographic, health, insurance)
- **Target Variables**: Annual cost, risk score, high-risk classification
- **Missing Values**: Handled via preprocessing pipeline (30% in `alcohol_freq`)

### Preprocessing Pipeline

- ✅ Missing value imputation (median/mode strategy)
- ✅ Standard scaling for numerical features
- ✅ One-hot encoding for categorical variables
- ✅ Unknown category handling for new data

### Model Architecture

- **Ensemble Methods**: Random Forest, Gradient Boosting, XGBoost
- **Evaluation**: Train/test split (80/20)
- **Cross-Validation**: Used for hyperparameter tuning
- **Serialization**: Joblib for efficient model storage

---

## 📦 Dependencies

```txt
pandas >= 2.0.0
numpy >= 1.24.0
scikit-learn >= 1.3.0
xgboost >= 2.0.0
streamlit >= 1.28.0
plotly >= 5.17.0
shap >= 0.42.0
reportlab >= 4.0.0
joblib >= 1.3.0
```

---

## 📚 Usage Examples

### Training Models

```python
# Train all models
python train_models.py

# Expected output:
# - Preprocessing pipeline created
# - Models trained and validated
# - Performance metrics saved
# - Models serialized to models/ directory
```

### Running Predictions

1. Launch dashboard: `streamlit run app.py`
2. Navigate to **Input** tab
3. Fill in patient information
4. Click **Predict** button
5. View results in **Results** tab

### Exporting Reports

1. Complete a prediction in the dashboard
2. Navigate to **Report** tab
3. Review underwriting summary
4. Download CSV or generate PDF report

---

## 🎓 Educational Use

This project is designed for educational and research purposes in:
- Machine Learning applications in actuarial science
- Healthcare data analysis
- Risk assessment and underwriting
- Insurance pricing models

---

## 📝 Notes

- Models are trained on historical data and predictions are estimates, not guarantees
- Results should be reviewed by qualified actuaries before making business decisions
- The system includes Indonesian language support for the user interface
- All currency values are in USD (United States Dollar)

---

## 👥 Contributing

This is an educational project. For questions or suggestions, please open an issue or contact the development team.

---

## 📄 License

This project is licensed for educational and research purposes only.

---

<div align="center">

**Built with ❤️ for Actuarial Science**

*Hybrid Actuarial Decision Support System v1.0*

</div>
