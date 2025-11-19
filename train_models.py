"""
Train machine learning models for healthcare insurance prediction.
Trains three models: cost regression, risk score regression, and high-risk classification.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix
)
import xgboost as xgb
import joblib
import os
from preprocessing import preprocess_data, create_preprocessing_pipeline

def train_cost_model(X_train, y_train, X_test, y_test):
    """Train regression models for annual medical cost."""
    print("\n" + "="*60)
    print("TRAINING COST REGRESSION MODELS")
    print("="*60)
    
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    }
    
    best_model = None
    best_score = float('inf')
    best_name = None
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Metrics
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        results[name] = {
            'model': model,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2
        }
        
        print(f"  Train MAE: {train_mae:.2f}, RMSE: {train_rmse:.2f}, R²: {train_r2:.4f}")
        print(f"  Test  MAE: {test_mae:.2f}, RMSE: {test_rmse:.2f}, R²: {test_r2:.4f}")
        
        if test_rmse < best_score:
            best_score = test_rmse
            best_model = model
            best_name = name
    
    print(f"\nBest model: {best_name} (Test RMSE: {best_score:.2f})")
    return best_model, results

def train_risk_score_model(X_train, y_train, X_test, y_test):
    """Train regression models for risk score."""
    print("\n" + "="*60)
    print("TRAINING RISK SCORE REGRESSION MODELS")
    print("="*60)
    
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1),
        'Linear Regression': LinearRegression()
    }
    
    best_model = None
    best_score = float('inf')
    best_name = None
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Metrics
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        results[name] = {
            'model': model,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2
        }
        
        print(f"  Train MAE: {train_mae:.4f}, RMSE: {train_rmse:.4f}, R²: {train_r2:.4f}")
        print(f"  Test  MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}")
        
        if test_rmse < best_score:
            best_score = test_rmse
            best_model = model
            best_name = name
    
    print(f"\nBest model: {best_name} (Test RMSE: {best_score:.4f})")
    return best_model, results

def train_high_risk_model(X_train, y_train, X_test, y_test):
    """Train classification models for high-risk prediction."""
    print("\n" + "="*60)
    print("TRAINING HIGH-RISK CLASSIFICATION MODELS")
    print("="*60)
    
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1),
        'XGBoost': xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    }
    
    best_model = None
    best_score = 0
    best_name = None
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        y_proba_train = model.predict_proba(X_train)[:, 1]
        y_proba_test = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        train_prec = precision_score(y_train, y_pred_train, zero_division=0)
        test_prec = precision_score(y_test, y_pred_test, zero_division=0)
        train_rec = recall_score(y_train, y_pred_train, zero_division=0)
        test_rec = recall_score(y_test, y_pred_test, zero_division=0)
        train_f1 = f1_score(y_train, y_pred_train, zero_division=0)
        test_f1 = f1_score(y_test, y_pred_test, zero_division=0)
        train_auc = roc_auc_score(y_train, y_proba_train)
        test_auc = roc_auc_score(y_test, y_proba_test)
        
        results[name] = {
            'model': model,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'train_prec': train_prec,
            'test_prec': test_prec,
            'train_rec': train_rec,
            'test_rec': test_rec,
            'train_f1': train_f1,
            'test_f1': test_f1,
            'train_auc': train_auc,
            'test_auc': test_auc
        }
        
        print(f"  Train Acc: {train_acc:.4f}, Prec: {train_prec:.4f}, Rec: {train_rec:.4f}, F1: {train_f1:.4f}, AUC: {train_auc:.4f}")
        print(f"  Test  Acc: {test_acc:.4f}, Prec: {test_prec:.4f}, Rec: {test_rec:.4f}, F1: {test_f1:.4f}, AUC: {test_auc:.4f}")
        
        if test_auc > best_score:
            best_score = test_auc
            best_model = model
            best_name = name
    
    print(f"\nBest model: {best_name} (Test AUC: {best_score:.4f})")
    return best_model, results

def save_models(cost_model, risk_model, high_risk_model, model_metrics=None):
    """Save trained models to disk."""
    os.makedirs('models', exist_ok=True)
    
    joblib.dump(cost_model, 'models/cost_model.pkl')
    print("\nSaved cost_model.pkl")
    
    joblib.dump(risk_model, 'models/risk_score_model.pkl')
    print("Saved risk_score_model.pkl")
    
    joblib.dump(high_risk_model, 'models/high_risk_model.pkl')
    print("Saved high_risk_model.pkl")
    
    if model_metrics:
        joblib.dump(model_metrics, 'models/model_metrics.pkl')
        print("Saved model_metrics.pkl")

def main():
    """Main training function."""
    print("="*60)
    print("HEALTHCARE INSURANCE PREDICTION MODEL TRAINING")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    df = pd.read_csv('medical_insurance.csv')
    print(f"Dataset shape: {df.shape}")
    
    # Create preprocessing pipeline
    print("\nCreating preprocessing pipeline...")
    preprocessor, metadata = create_preprocessing_pipeline(df)
    
    # Preprocess data
    print("\nPreprocessing data...")
    X, y, metadata = preprocess_data(df, preprocessor)
    
    # Split data
    print("\nSplitting data into train/test sets...")
    X_train, X_test, y_train_cost, y_test_cost = train_test_split(
        X, y['cost'], test_size=0.2, random_state=42
    )
    _, _, y_train_risk, y_test_risk = train_test_split(
        X, y['risk_score'], test_size=0.2, random_state=42
    )
    _, _, y_train_high_risk, y_test_high_risk = train_test_split(
        X, y['is_high_risk'], test_size=0.2, random_state=42
    )
    
    print(f"Train set: {X_train.shape[0]:,} samples")
    print(f"Test set: {X_test.shape[0]:,} samples")
    
    # Train models
    cost_model, cost_results = train_cost_model(X_train, y_train_cost, X_test, y_test_cost)
    risk_model, risk_results = train_risk_score_model(X_train, y_train_risk, X_test, y_test_risk)
    high_risk_model, high_risk_results = train_high_risk_model(
        X_train, y_train_high_risk, X_test, y_test_high_risk
    )
    
    # Save models
    print("\n" + "="*60)
    print("SAVING MODELS")
    print("="*60)
    model_metrics = {
        'cost': cost_results,
        'risk_score': risk_results,
        'high_risk': high_risk_results
    }
    save_models(cost_model, risk_model, high_risk_model, model_metrics)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()


