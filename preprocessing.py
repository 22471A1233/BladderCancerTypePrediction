import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import os

def clean_data(df):
    """
    Handle missing values and drop columns with >40% missing values
    """
    # 1. Drop columns with >40% missing values
    missing_ratio = df.isnull().mean()
    drop_cols = missing_ratio[missing_ratio > 0.4].index.tolist()
    df = df.drop(columns=drop_cols)
    print(f"Dropped columns with >40% missing values: {drop_cols}")
    
    # 2. Median imputation for numerical columns
    num_cols = df.select_dtypes(include=[np.number]).columns
    imputer = SimpleImputer(strategy='median')
    df[num_cols] = imputer.fit_transform(df[num_cols])
    
    # 3. Mode imputation for categorical columns
    cat_cols = df.select_dtypes(exclude=[np.number]).columns
    if len(cat_cols) > 0:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])
        
    return df

def encode_categorical(df, encoders=None):
    """
    Encode categorical features using LabelEncoder
    """
    df = df.copy()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    if 'Target' in cat_cols:
        cat_cols.remove('Target')
        
    if encoders is None:
        encoders = {}
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    else:
        for col, le in encoders.items():
            if col in df.columns:
                # Handle unknown labels by assigning to most frequent (mode)
                df[col] = df[col].astype(str).map(lambda x: x if x in le.classes_ else le.classes_[0])
                df[col] = le.transform(df[col])
                
    return df, encoders

def scale_features(X, scaler=None):
    """
    Normalize features using StandardScaler
    """
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
        
    return X_scaled, scaler
