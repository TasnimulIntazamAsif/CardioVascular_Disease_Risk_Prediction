#!/usr/bin/env python3
"""
Test multiple dataset samples to find mismatches
"""

import pandas as pd
import numpy as np
import joblib
import requests
import json

def test_multiple_samples():
    """Test multiple samples from the dataset"""
    try:
        df = pd.read_csv('CVD Dataset.csv')
        df = df.dropna(subset=['CVD Risk Level'])
        
        # Load model and parameters
        model = joblib.load('models/rf_weighted_fusion_model.joblib')
        scaler_clinical = joblib.load('models/scaler_clinical.joblib')
        scaler_lifestyle = joblib.load('models/scaler_lifestyle.joblib')
        fusion_params = joblib.load('models/fusion_params.joblib')
        
        mismatches = []
        matches = []
        
        # Test first 10 samples
        for i in range(min(10, len(df))):
            sample_row = df.iloc[i]
            
            # Convert to Flask app format with NaN handling
            def safe_float(value):
                try:
                    if pd.isna(value) or np.isnan(value):
                        return 0.0
                    return float(value)
                except (ValueError, TypeError):
                    return 0.0
            
            def safe_str(value):
                try:
                    if pd.isna(value):
                        return 'N'  # Default value
                    return str(value)
                except (ValueError, TypeError):
                    return 'N'
            
            flask_data = {
                'Age': safe_float(sample_row['Age']),
                'Sex': safe_str(sample_row['Sex']),
                'Weight (kg)': safe_float(sample_row['Weight (kg)']),
                'Height (m)': safe_float(sample_row['Height (m)']),
                'Systolic BP': safe_float(sample_row['Systolic BP']),
                'Diastolic BP': safe_float(sample_row['Diastolic BP']),
                'Smoking Status': safe_str(sample_row['Smoking Status']),
                'Diabetes Status': safe_str(sample_row['Diabetes Status']),
                'Physical Activity Level': safe_str(sample_row['Physical Activity Level']),
                'Family History of CVD': safe_str(sample_row['Family History of CVD']),
                'Height (cm)': safe_float(sample_row['Height (cm)']),
                'Abdominal Circumference (cm)': safe_float(sample_row['Abdominal Circumference (cm)']),
                'Total Cholesterol (mg/dL)': safe_float(sample_row['Total Cholesterol (mg/dL)']),
                'HDL (mg/dL)': safe_float(sample_row['HDL (mg/dL)']),
                'Fasting Blood Sugar (mg/dL)': safe_float(sample_row['Fasting Blood Sugar (mg/dL)']),
                'Estimated LDL (mg/dL)': safe_float(sample_row['Estimated LDL (mg/dL)'])
            }
            
            expected_risk = sample_row['CVD Risk Level']
            
            # Test direct prediction
            direct_risk = test_direct_prediction(flask_data, model, scaler_clinical, scaler_lifestyle, fusion_params)
            
            # Test Flask prediction
            flask_risk = test_flask_prediction(flask_data)
            
            print(f"\nSample {i+1}:")
            print(f"Expected: {expected_risk}")
            print(f"Direct: {direct_risk}")
            print(f"Flask: {flask_risk}")
            
            if direct_risk == flask_risk:
                matches.append(i+1)
                print("MATCH")
            else:
                mismatches.append(i+1)
                print("MISMATCH!")
        
        print(f"\n" + "="*50)
        print("SUMMARY:")
        print("="*50)
        print(f"Matches: {len(matches)} samples")
        print(f"Mismatches: {len(mismatches)} samples")
        
        if mismatches:
            print(f"Mismatched samples: {mismatches}")
        else:
            print("All samples matched!")
            
        return len(mismatches) == 0
        
    except Exception as e:
        print(f"Error in testing: {e}")
        return False

def test_direct_prediction(data, model, scaler_clinical, scaler_lifestyle, fusion_params):
    """Test direct prediction using the model"""
    try:
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
        
        # Map to risk levels
        risk_mapping = {0: 'LOW', 1: 'INTERMEDIARY', 2: 'HIGH'}
        risk_level = risk_mapping.get(prediction, 'UNKNOWN')
        
        return risk_level
        
    except Exception as e:
        print(f"Error in direct prediction: {e}")
        return None

def test_flask_prediction(data):
    """Test Flask app prediction"""
    try:
        response = requests.post('http://localhost:5000/predict', json=data)
        
        if response.status_code == 200:
            result = response.json()
            return result['risk_level']
        else:
            print(f"Flask app error: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"Error in Flask prediction: {e}")
        return None

if __name__ == "__main__":
    print("=" * 80)
    print("TESTING MULTIPLE DATASET SAMPLES")
    print("=" * 80)
    
    success = test_multiple_samples()
    
    if success:
        print("\nAll samples matched! Flask app is working correctly.")
    else:
        print("\nSome samples had mismatches. Need to investigate further.")
