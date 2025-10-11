#!/usr/bin/env python3
"""
Test the existing weighted fusion model to check its performance
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler

def test_existing_model():
    """Test the existing weighted fusion model"""
    try:
        # Load the model
        model = joblib.load('models/rf_weighted_fusion_model.joblib')
        scaler_clinical = joblib.load('models/scaler_clinical.joblib')
        scaler_lifestyle = joblib.load('models/scaler_lifestyle.joblib')
        fusion_params = joblib.load('models/fusion_params.joblib')
        
        print("Model loaded successfully!")
        print(f"Model type: {type(model)}")
        print(f"Fusion params: {fusion_params}")
        
        # Load and preprocess data
        df = pd.read_csv('CVD Dataset.csv')
        df = df.dropna(subset=['CVD Risk Level'])
        
        # Create label encoder for target
        label_encoder = LabelEncoder()
        df['CVD Risk Level_encoded'] = label_encoder.fit_transform(df['CVD Risk Level'])
        
        # Handle missing values in features
        numeric_features = ['Age', 'Weight (kg)', 'Height (m)', 'Systolic BP', 'Diastolic BP', 
                          'Total Cholesterol (mg/dL)', 'HDL (mg/dL)', 'Fasting Blood Sugar (mg/dL)', 
                          'Estimated LDL (mg/dL)', 'Abdominal Circumference (cm)', 'Height (cm)']
        
        for feature in numeric_features:
            if feature in df.columns:
                df[feature] = df[feature].fillna(df[feature].median())
        
        # Encode categorical features
        categorical_features = ['Sex', 'Smoking Status', 'Diabetes Status', 'Physical Activity Level', 
                              'Family History of CVD', 'Blood Pressure Category']
        
        for feature in categorical_features:
            if feature in df.columns:
                le = LabelEncoder()
                df[f'{feature}_encoded'] = le.fit_transform(df[feature].astype(str))
        
        # Calculate derived features
        df['BMI'] = df['Weight (kg)'] / (df['Height (m)'] ** 2)
        df['Waist-to-Height Ratio'] = df['Abdominal Circumference (cm)'] / df['Height (cm)']
        
        # Calculate CVD Risk Score
        df['CVD Risk Score'] = (
            df['Age'] * 0.1 + 
            df['BMI'] * 0.2 + 
            df['Systolic BP'] * 0.15 + 
            df['Total Cholesterol (mg/dL)'] * 0.1 +
            df['Fasting Blood Sugar (mg/dL)'] * 0.1 +
            df['Smoking Status_encoded'] * 2 +
            df['Diabetes Status_encoded'] * 1.5 +
            df['Family History of CVD_encoded'] * 1
        )
        
        # Prepare feature data
        clinical_features = fusion_params['clinical_features']
        lifestyle_features = fusion_params['lifestyle_features']
        
        clinical_data = df[clinical_features].values
        lifestyle_data = df[lifestyle_features].values
        
        # Scale features separately
        clinical_scaled = scaler_clinical.transform(clinical_data)
        lifestyle_scaled = scaler_lifestyle.transform(lifestyle_data)
        
        # Apply weighted fusion
        weight_clinical = fusion_params['weight_clinical']
        weight_lifestyle = fusion_params['weight_lifestyle']
        
        clinical_weighted = clinical_scaled * weight_clinical
        lifestyle_weighted = lifestyle_scaled * weight_lifestyle
        
        # Concatenate weighted features
        X_fused = np.hstack([clinical_weighted, lifestyle_weighted])
        y = df['CVD Risk Level_encoded'].values
        
        print(f"Fused features shape: {X_fused.shape}")
        print(f"Target distribution: {np.bincount(y)}")
        
        # Test the model
        y_pred = model.predict(X_fused)
        accuracy = accuracy_score(y, y_pred)
        print(f"Model accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y, y_pred))
        
        return True
        
    except Exception as e:
        print(f"Error testing model: {e}")
        return False

if __name__ == "__main__":
    print("=" * 80)
    print("TESTING EXISTING WEIGHTED FUSION MODEL")
    print("=" * 80)
    
    success = test_existing_model()
    
    if success:
        print("\n✅ Model test completed successfully!")
    else:
        print("\n❌ Model test failed!")
