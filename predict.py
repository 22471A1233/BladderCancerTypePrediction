import pickle
import pandas as pd
import numpy as np
import os
import sys

# Import from utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.preprocessing import scale_features

class BladderCancerPredictor:
    def __init__(self, model_dir='model'):
        self.model_path = os.path.join(model_dir, 'model.pkl')
        self.scaler_path = os.path.join(model_dir, 'scaler.pkl')
        self.features_path = os.path.join(model_dir, 'features.pkl')
        self.target_encoder_path = os.path.join(model_dir, 'target_encoder.pkl')
        self.feature_encoders_path = os.path.join(model_dir, 'feature_encoders.pkl')
        
        self.load_artifacts()
        
    def load_artifacts(self):
        if not all(os.path.exists(p) for p in [self.model_path, self.scaler_path, self.features_path]):
            raise FileNotFoundError("Model artifacts not found. Please run training first.")
            
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)
            
        with open(self.scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
            
        with open(self.features_path, 'rb') as f:
            self.selected_features = pickle.load(f)
            
        with open(self.target_encoder_path, 'rb') as f:
            self.target_encoder = pickle.load(f)
            
        with open(self.feature_encoders_path, 'rb') as f:
            self.feature_encoders = pickle.load(f)
            
    def predict(self, input_data):
        """
        input_data: dict containing feature names and values
        """
        # Convert to DataFrame
        df = pd.DataFrame([input_data])
        
        # 1. Encode categorical features if any in selected features
        for col, le in self.feature_encoders.items():
            if col in df.columns:
                # Handle unknown labels
                val = str(df[col].iloc[0])
                if val in le.classes_:
                    df[col] = le.transform([val])[0]
                else:
                    df[col] = le.transform([le.classes_[0]])[0]
        
        # 2. Ensure all selected features are present and in correct order
        # This is critical for model consistency
        X = df[self.selected_features]
        
        # 3. Scale using the loaded scaler
        X_scaled = self.scaler.transform(X)
        
        # 4. Predict probabilities
        probabilities = self.model.predict_proba(X_scaled)[0]
        prediction_idx = np.argmax(probabilities)
        
        # 5. Get labels and confidence
        prediction_label = self.target_encoder.inverse_transform([prediction_idx])[0]
        class_labels = self.target_encoder.classes_
        
        # Confidence score as percentage (0-100)
        confidence = float(np.max(probabilities) * 100)
        
        # Determine risk level based on confidence score
        if confidence > 80:
            risk_level = "High"
            risk_color = "Red"
        elif confidence > 50:
            risk_level = "Medium"
            risk_color = "Yellow"
        else:
            risk_level = "Low"
            risk_color = "Green"
            
        return {
            'prediction': prediction_label,
            'confidence': confidence,
            'risk_level': risk_level,
            'risk_color': risk_color,
            'probabilities': probabilities.tolist(),
            'classes': class_labels.tolist()
        }
