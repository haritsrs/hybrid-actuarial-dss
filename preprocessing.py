"""
Data preprocessing pipeline for medical insurance dataset.
Handles missing values, encoding, scaling, and feature preparation.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import os

def create_preprocessing_pipeline(df, save_path='models/preprocessor.pkl'):
    """
    Create and fit preprocessing pipeline.
    
    Args:
        df: DataFrame with raw data
        save_path: Path to save the preprocessor
    
    Returns:
        Fitted preprocessing pipeline
    """
    # Define feature columns (exclude targets and person_id)
    target_cols = ['annual_medical_cost', 'risk_score', 'is_high_risk']
    exclude_cols = ['person_id'] + target_cols
    
    # Separate categorical and numerical features
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove excluded columns
    categorical_cols = [col for col in categorical_cols if col not in exclude_cols]
    numerical_cols = [col for col in numerical_cols if col not in exclude_cols]
    
    # Create preprocessing pipelines
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])
    
    # Combine pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_cols),
            ('cat', categorical_pipeline, categorical_cols)
        ],
        remainder='passthrough'
    )
    
    # Fit the preprocessor
    feature_cols = numerical_cols + categorical_cols
    X = df[feature_cols]
    preprocessor.fit(X)
    
    # Save preprocessor
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(preprocessor, save_path)
    print(f"Preprocessor saved to {save_path}")
    
    # Save feature names for later use
    feature_names = numerical_cols.copy()
    # Get one-hot encoded feature names
    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
    cat_feature_names = cat_encoder.get_feature_names_out(categorical_cols)
    feature_names.extend(cat_feature_names)
    
    metadata = {
        'numerical_cols': numerical_cols,
        'categorical_cols': categorical_cols,
        'feature_names': feature_names
    }
    
    metadata_path = save_path.replace('.pkl', '_metadata.pkl')
    joblib.dump(metadata, metadata_path)
    print(f"Preprocessor metadata saved to {metadata_path}")
    
    return preprocessor, metadata

def preprocess_data(df, preprocessor=None, preprocessor_path='models/preprocessor.pkl'):
    """
    Preprocess the dataset.
    
    Args:
        df: Raw DataFrame
        preprocessor: Fitted preprocessor (if None, loads from path)
        preprocessor_path: Path to saved preprocessor
    
    Returns:
        X: Preprocessed features
        y: Target variables
        metadata: Preprocessor metadata
    """
    if preprocessor is None:
        preprocessor = joblib.load(preprocessor_path)
        metadata = joblib.load(preprocessor_path.replace('.pkl', '_metadata.pkl'))
    else:
        metadata = joblib.load(preprocessor_path.replace('.pkl', '_metadata.pkl'))
    
    # Get feature columns
    feature_cols = metadata['numerical_cols'] + metadata['categorical_cols']
    
    # Prepare features and targets
    X = df[feature_cols]
    y = {
        'cost': df['annual_medical_cost'].values,
        'risk_score': df['risk_score'].values,
        'is_high_risk': df['is_high_risk'].values.astype(int)
    }
    
    # Transform features
    X_transformed = preprocessor.transform(X)
    
    return X_transformed, y, metadata

if __name__ == "__main__":
    # Load data
    print("Loading data...")
    df = pd.read_csv('medical_insurance.csv')
    
    # Create and save preprocessor
    print("Creating preprocessing pipeline...")
    preprocessor, metadata = create_preprocessing_pipeline(df)
    
    # Test preprocessing
    print("Testing preprocessing...")
    X, y, metadata = preprocess_data(df, preprocessor)
    
    print(f"\nPreprocessed shape: {X.shape}")
    print(f"Target shapes:")
    print(f"  Cost: {y['cost'].shape}")
    print(f"  Risk Score: {y['risk_score'].shape}")
    print(f"  High Risk: {y['is_high_risk'].shape}")

