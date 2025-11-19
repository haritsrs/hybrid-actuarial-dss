"""
Data exploration script for medical insurance dataset.
Analyzes dataset structure, missing values, distributions, and correlations.
"""

import pandas as pd
import numpy as np

def explore_data(file_path='medical_insurance.csv'):
    """Load and explore the medical insurance dataset."""
    print("Loading dataset...")
    df = pd.read_csv(file_path)
    
    print(f"\n{'='*60}")
    print("DATASET OVERVIEW")
    print(f"{'='*60}")
    print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"\nColumn names:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2d}. {col}")
    
    print(f"\n{'='*60}")
    print("DATA TYPES")
    print(f"{'='*60}")
    print(df.dtypes.value_counts())
    
    print(f"\n{'='*60}")
    print("MISSING VALUES")
    print(f"{'='*60}")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Missing %': missing_pct
    })
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
    if len(missing_df) > 0:
        print(missing_df)
    else:
        print("No missing values found!")
    
    print(f"\n{'='*60}")
    print("DUPLICATE ROWS")
    print(f"{'='*60}")
    duplicates = df.duplicated().sum()
    print(f"Duplicate rows: {duplicates}")
    
    print(f"\n{'='*60}")
    print("TARGET VARIABLES SUMMARY")
    print(f"{'='*60}")
    targets = ['annual_medical_cost', 'risk_score', 'is_high_risk']
    for target in targets:
        if target in df.columns:
            print(f"\n{target}:")
            print(df[target].describe())
            if target == 'is_high_risk':
                print(f"\nValue counts:")
                print(df[target].value_counts())
                print(f"\nProportions:")
                print(df[target].value_counts(normalize=True))
    
    print(f"\n{'='*60}")
    print("CATEGORICAL FEATURES")
    print(f"{'='*60}")
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    print(f"Categorical columns ({len(categorical_cols)}):")
    for col in categorical_cols:
        unique_count = df[col].nunique()
        print(f"  {col}: {unique_count} unique values")
        if unique_count <= 10:
            print(f"    Values: {df[col].unique().tolist()}")
    
    print(f"\n{'='*60}")
    print("NUMERICAL FEATURES SUMMARY")
    print(f"{'='*60}")
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude target variables from numerical summary
    numerical_cols = [col for col in numerical_cols if col not in targets and col != 'person_id']
    print(f"Numerical columns ({len(numerical_cols)}):")
    print(df[numerical_cols].describe())
    
    print(f"\n{'='*60}")
    print("CORRELATION WITH TARGETS")
    print(f"{'='*60}")
    for target in targets:
        if target in df.columns:
            print(f"\nCorrelation with {target}:")
            correlations = df[numerical_cols + [target]].corr()[target].sort_values(ascending=False)
            correlations = correlations[correlations.index != target]
            print(correlations.head(10))
            print("\nTop negative correlations:")
            print(correlations.tail(10))
    
    return df

if __name__ == "__main__":
    df = explore_data()


