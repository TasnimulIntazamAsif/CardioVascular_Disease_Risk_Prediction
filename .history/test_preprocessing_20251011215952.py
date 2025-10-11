#!/usr/bin/env python3
"""
Test the Flask app preprocessing to ensure it matches the notebook implementation
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

def test_preprocessing():
    """Test the preprocessing pipeline"""
    try:
        # Load the actual model and parameters
        model = joblib.load('models/rf_weighted_fusion_model.joblib')
        scaler_clinical = joblib.load('models/scaler_clinical.joblib')
        scaler_lifestyle = joblib.load('models/scaler_lifestyle.joblib')
        fusion_params = joblib.load('models/fusion_params.joblib')
        
        print("Model and parameters loaded successfully!")
        print(f"Model type: {type(model)}")
        print(f"Fusion params: {fusion_params}")
        
        # Test with sample data
        test_data = {
            'Age': 45.0,
            'Sex': 'M',
            'Weight (kg)': 80.0,
            'Height (m)': 1.75,
            'Systolic BP': 130.0,
            'Diastolic BP': 85.0,
            'Smoking Status': 'N',
            'Diabetes Status': 'N',
            'Physical Activity Level': 'Moderate',
            'Family History of CVD': 'N',
            'Height (cm)': 175.0,
            'Abdominal Circumference (cm)': 90.0,
            'Total Cholesterol (mg/dL)': 200.0,
            'HDL (mg/dL)': 50.0,
            'Fasting Blood Sugar (mg/dL)': 100.0,
            'Estimated LDL (mg/dL)': 120.0
        }
        
        print(f"\nTest data: {test_data}")
        
        # Convert to DataFrame
        df = pd.DataFrame([test_data])
        
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
        
        print(f"\nProcessed DataFrame:")
        print(df.head())
        
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
        
        print(f"\nClinical features: {clinical_features}")
        print(f"Clinical data: {clinical_data}")
        print(f"Lifestyle features: {lifestyle_features}")
        print(f"Lifestyle data: {lifestyle_data}")
        
        # Convert to numpy arrays
        X_clinical = np.array(clinical_data).reshape(1, -1)
        X_lifestyle = np.array(lifestyle_data).reshape(1, -1)
        
        print(f"\nClinical array shape: {X_clinical.shape}")
        print(f"Lifestyle array shape: {X_lifestyle.shape}")
        
        # Scale the features separately
        Xc_scaled = scaler_clinical.transform(X_clinical)
        Xl_scaled = scaler_lifestyle.transform(X_lifestyle)
        
        print(f"Clinical scaled shape: {Xc_scaled.shape}")
        print(f"Lifestyle scaled shape: {Xl_scaled.shape}")
        
        # Apply weighted fusion using optimal weights
        weight_clinical = fusion_params['weight_clinical']
        weight_lifestyle = fusion_params['weight_lifestyle']
        
        Xc_weighted = Xc_scaled * weight_clinical
        Xl_weighted = Xl_scaled * weight_lifestyle
        
        # Concatenate weighted features
        X_fused = np.hstack([Xc_weighted, Xl_weighted])
        
        print(f"\nFinal fused features shape: {X_fused.shape}")
        print(f"Weights - Clinical: {weight_clinical}, Lifestyle: {weight_lifestyle}")
        
        # Make prediction
        prediction = model.predict(X_fused)
        probabilities = model.predict_proba(X_fused)
        
        print(f"\nPrediction: {prediction[0]}")
        print(f"Probabilities: {probabilities[0]}")
        
        # Map to risk levels
        risk_mapping = {0: 'LOW', 1: 'INTERMEDIARY', 2: 'HIGH'}
        risk_level = risk_mapping.get(prediction[0], 'UNKNOWN')
        confidence = max(probabilities[0]) * 100
        
        print(f"\nRisk Level: {risk_level}")
        print(f"Confidence: {confidence:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"Error in testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 80)
    print("TESTING PREPROCESSING PIPELINE")
    print("=" * 80)
    
    success = test_preprocessing()
    
    if success:
        print("\nPreprocessing test completed successfully!")
    else:
        print("\nPreprocessing test failed!")
