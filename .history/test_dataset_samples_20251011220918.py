#!/usr/bin/env python3
"""
Test Flask app with actual dataset samples to verify predictions match
"""

import pandas as pd
import numpy as np
import joblib
import requests
import json

def load_dataset_sample():
    """Load a sample from the actual dataset"""
    try:
        df = pd.read_csv('CVD Dataset.csv')
        df = df.dropna(subset=['CVD Risk Level'])
        
        # Get a sample row
        sample_row = df.iloc[0]  # First row
        print(f"Dataset sample (row 0):")
        print(f"CVD Risk Level: {sample_row['CVD Risk Level']}")
        
        # Convert to Flask app format
        flask_data = {
            'Age': float(sample_row['Age']),
            'Sex': str(sample_row['Sex']),
            'Weight (kg)': float(sample_row['Weight (kg)']),
            'Height (m)': float(sample_row['Height (m)']),
            'Systolic BP': float(sample_row['Systolic BP']),
            'Diastolic BP': float(sample_row['Diastolic BP']),
            'Smoking Status': str(sample_row['Smoking Status']),
            'Diabetes Status': str(sample_row['Diabetes Status']),
            'Physical Activity Level': str(sample_row['Physical Activity Level']),
            'Family History of CVD': str(sample_row['Family History of CVD']),
            'Height (cm)': float(sample_row['Height (cm)']),
            'Abdominal Circumference (cm)': float(sample_row['Abdominal Circumference (cm)']),
            'Total Cholesterol (mg/dL)': float(sample_row['Total Cholesterol (mg/dL)']),
            'HDL (mg/dL)': float(sample_row['HDL (mg/dL)']),
            'Fasting Blood Sugar (mg/dL)': float(sample_row['Fasting Blood Sugar (mg/dL)']),
            'Estimated LDL (mg/dL)': float(sample_row['Estimated LDL (mg/dL)'])
        }
        
        return flask_data, sample_row['CVD Risk Level']
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None

def test_direct_prediction(data):
    """Test direct prediction using the model"""
    try:
        # Load model and parameters
        model = joblib.load('models/rf_weighted_fusion_model.joblib')
        scaler_clinical = joblib.load('models/scaler_clinical.joblib')
        scaler_lifestyle = joblib.load('models/scaler_lifestyle.joblib')
        fusion_params = joblib.load('models/fusion_params.joblib')
        
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Handle categorical features
        categorical_features = ['Sex', 'Smoking Status', 'Diabetes Status', 'Physical Activity Level', 'Family History of CVD']
        
        for feature in categorical_features:
            if feature in df.columns:
                if feature == 'Sex':
                    df[feature] = df[feature].map({'M': 1, 'F': 0}).fillna(0)
                elif feature == 'Smoking Status':
                    df[feature] = df[feature].map({'Y': 1, 'N': 0}).fillna(0)
                elif feature == 'Diabetes Status':
                    df[feature] = df[feature].map({'Y': 1, 'N': 0}).fillna(0)
                elif feature == 'Physical Activity Level':
                    df[feature] = df[feature].map({'High': 2, 'Moderate': 1, 'Low': 0}).fillna(0)
                elif feature == 'Family History of CVD':
                    df[feature] = df[feature].map({'Y': 1, 'N': 0}).fillna(0)
        
        # Calculate derived features
        df['BMI'] = df['Weight (kg)'] / (df['Height (m)'] ** 2)
        df['Waist-to-Height Ratio'] = df['Abdominal Circumference (cm)'] / df['Height (cm)']
        
        # Prepare clinical features
        clinical_features = fusion_params['clinical_features']
        clinical_data = []
        for feature in clinical_features:
            if feature in df.columns:
                clinical_data.append(df[feature].iloc[0])
            else:
                clinical_data.append(0.0)
        
        # Prepare lifestyle features
        lifestyle_features = fusion_params['lifestyle_features']
        lifestyle_data = []
        for feature in lifestyle_features:
            if feature in df.columns:
                lifestyle_data.append(df[feature].iloc[0])
            else:
                lifestyle_data.append(0.0)
        
        # Convert to numpy arrays
        X_clinical = np.array(clinical_data).reshape(1, -1)
        X_lifestyle = np.array(lifestyle_data).reshape(1, -1)
        
        # Scale the features separately
        Xc_scaled = scaler_clinical.transform(X_clinical)
        Xl_scaled = scaler_lifestyle.transform(X_lifestyle)
        
        # Apply weighted fusion using optimal weights
        weight_clinical = fusion_params['weight_clinical']
        weight_lifestyle = fusion_params['weight_lifestyle']
        
        Xc_weighted = Xc_scaled * weight_clinical
        Xl_weighted = Xl_scaled * weight_lifestyle
        
        # Concatenate weighted features
        X_fused = np.hstack([Xc_weighted, Xl_weighted])
        
        # Make prediction
        prediction = model.predict(X_fused)[0]
        probabilities = model.predict_proba(X_fused)[0]
        
        # Map to risk levels
        risk_mapping = {0: 'LOW', 1: 'INTERMEDIARY', 2: 'HIGH'}
        risk_level = risk_mapping.get(prediction, 'UNKNOWN')
        confidence = max(probabilities) * 100
        
        return risk_level, confidence, prediction, probabilities
        
    except Exception as e:
        print(f"Error in direct prediction: {e}")
        return None, None, None, None

def test_flask_prediction(data):
    """Test Flask app prediction"""
    try:
        response = requests.post('http://localhost:5000/predict', json=data)
        
        if response.status_code == 200:
            result = response.json()
            return result['risk_level'], result['confidence'], None, None
        else:
            print(f"Flask app error: {response.status_code}")
            return None, None, None, None
            
    except Exception as e:
        print(f"Error in Flask prediction: {e}")
        return None, None, None, None

if __name__ == "__main__":
    print("=" * 80)
    print("TESTING WITH ACTUAL DATASET SAMPLES")
    print("=" * 80)
    
    # Load dataset sample
    flask_data, expected_risk = load_dataset_sample()
    
    if flask_data is None:
        print("Could not load dataset sample")
        exit(1)
    
    print(f"\nExpected CVD Risk Level: {expected_risk}")
    print(f"Sample data: {flask_data}")
    
    # Test direct prediction
    print(f"\n" + "="*50)
    print("DIRECT MODEL PREDICTION:")
    print("="*50)
    direct_risk, direct_conf, direct_pred, direct_probs = test_direct_prediction(flask_data)
    
    if direct_risk:
        print(f"Predicted Risk Level: {direct_risk}")
        print(f"Confidence: {direct_conf:.1f}%")
        print(f"Raw Prediction: {direct_pred}")
        print(f"Probabilities: {direct_probs}")
    
    # Test Flask prediction
    print(f"\n" + "="*50)
    print("FLASK APP PREDICTION:")
    print("="*50)
    flask_risk, flask_conf, _, _ = test_flask_prediction(flask_data)
    
    if flask_risk:
        print(f"Predicted Risk Level: {flask_risk}")
        print(f"Confidence: {flask_conf:.1f}%")
    
    # Compare results
    print(f"\n" + "="*50)
    print("COMPARISON:")
    print("="*50)
    
    if direct_risk and flask_risk:
        print(f"Expected: {expected_risk}")
        print(f"Direct Model: {direct_risk}")
        print(f"Flask App: {flask_risk}")
        
        if direct_risk == flask_risk:
            print("✅ Direct model and Flask app predictions match!")
        else:
            print("❌ Direct model and Flask app predictions differ!")
            
        if direct_risk == expected_risk:
            print("✅ Direct model matches expected result!")
        else:
            print("❌ Direct model does not match expected result!")
            
        if flask_risk == expected_risk:
            print("✅ Flask app matches expected result!")
        else:
            print("❌ Flask app does not match expected result!")
    else:
        print("❌ Could not compare results due to errors.")
