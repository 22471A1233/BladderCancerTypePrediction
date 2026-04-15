import pandas as pd
import numpy as np
import os
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

# Import from utils
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.preprocessing import clean_data, encode_categorical, scale_features

def train_model():
    print("Starting ML Pipeline Training...")
    
    # 1. Load Data
    data_path = 'data/dataset.csv'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return
    
    df = pd.read_csv(data_path)
    
    # 2. Clean Data (Median imputation, Drop >40% missing)
    df = clean_data(df)
    
    # 3. Encode Categorical Features
    # Separate features and target
    X = df.drop(columns=['Target'])
    y = df['Target']
    
    # Encode target labels
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)
    classes = le_target.classes_.tolist()
    
    # Encode categorical features
    X_encoded, encoders = encode_categorical(X)
    
    # 4. Feature Selection (Top 12-15 using XGBoost Importance)
    print("Selecting top features...")
    xgb_selector = XGBClassifier(random_state=42)
    xgb_selector.fit(X_encoded, y_encoded)
    
    importances = xgb_selector.feature_importances_
    indices = np.argsort(importances)[-15:] # Top 15
    selected_features = X_encoded.columns[indices].tolist()
    
    X_selected = X_encoded[selected_features]
    print(f"Selected features: {selected_features}")
    
    # 5. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    
    # 6. SMOTE for Class Imbalance
    print("Balancing classes with SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    # 7. Scaling
    X_train_scaled, scaler = scale_features(X_train_balanced)
    X_test_scaled, _ = scale_features(X_test, scaler)
    
    # 8. Model Training & Hyperparameter Tuning (XGBoost)
    print("Tuning XGBoost Hyperparameters...")
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 4, 5, 6],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'gamma': [0, 0.1, 0.2]
    }
    
    xgb_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')
    random_search = RandomizedSearchCV(
        xgb_model, 
        param_distributions=param_grid, 
        n_iter=15, 
        scoring='f1_weighted', 
        cv=5, 
        random_state=42,
        n_jobs=-1
    )
    random_search.fit(X_train_scaled, y_train_balanced)
    
    best_model = random_search.best_estimator_
    print(f"Best XGBoost Params: {random_search.best_params_}")
    
    # 9. Evaluate Best Model
    y_pred = best_model.predict(X_test_scaled)
    y_prob = best_model.predict_proba(X_test_scaled)
    
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted'),
        'Recall': recall_score(y_test, y_pred, average='weighted'),
        'F1-Score': f1_score(y_test, y_pred, average='weighted'),
        'ROC-AUC': roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted')
    }
    
    print("\nModel Evaluation Metrics:")
    for m, val in metrics.items():
        print(f"{m}: {val:.4f}")
        
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=classes))
    
    # 10. Save Artifacts
    os.makedirs('model', exist_ok=True)
    
    with open('model/model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
        
    with open('model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
        
    with open('model/features.pkl', 'wb') as f:
        pickle.dump(selected_features, f)
        
    with open('model/target_encoder.pkl', 'wb') as f:
        pickle.dump(le_target, f)
        
    with open('model/feature_encoders.pkl', 'wb') as f:
        # We need to save the individual encoders for categorical features
        # Filter for only encoders that are in selected features
        selected_cat_encoders = {k: v for k, v in encoders.items() if k in selected_features}
        pickle.dump(selected_cat_encoders, f)
        
    # Save metrics for About page
    with open('model/metrics.json', 'w') as f:
        json.dump(metrics, f)
        
    print("Training complete! Artifacts saved in 'model/' directory.")

if __name__ == "__main__":
    train_model()
