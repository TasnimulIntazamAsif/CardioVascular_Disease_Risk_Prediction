#!/usr/bin/env python3
"""
Test the model directly to see if it's the issue
"""

import pandas as pd
import numpy as np
import joblib

def test_model_directly():
    """Test the model directly without Flask app"""
    try:
        # Load model and parameters
        model = joblib.load('models/rf_weighted_fusion_model.joblib')
        scaler_clinical = joblib.load('models/scaler_clinical.joblib')
        scaler_lifestyle = joblib.load('models/scaler_lifestyle.joblib')
        fusion_params = joblib.load('models/fusion_params.joblib')
        
        print("Model loaded successfully!")
        print(f"Model type: {type(model)}")
        print(f"Fusion params: {fusion_params}")
        
        # Load dataset
        df = pd.read_csv('CVD Dataset.csv')
        df = df.dropna(subset=['CVD Risk Level'])
        
        print(f"\nDataset shape: {df.shape}")
        print("Dataset CVD Risk Level distribution:")
        print(df['CVD Risk Level'].value_counts())
        
        # Test first 20 samples
        predictions = []
        expected = []
        
        for i in range(min(20, len(df))):
            sample_row = df.iloc[i]
            
            # Convert to DataFrame
            data = {
                'Age': float(sample_row['Age']) if not pd.isna(sample_row['Age']) else 0.0,
                'Sex': str(sample_row['Sex']) if not pd.isna(sample_row['Sex']) else 'N',
                'Weight (kg)': float(sample_row['Weight (kg)']) if not pd.isna(sample_row['Weight (kg)']) else 0.0,
                'Height (m)': float(sample_row['Height (m)']) if not pd.isna(sample_row['Height (m)']) else 0.0,
                'Systolic BP': float(sample_row['Systolic BP']) if not pd.isna(sample_row['Systolic BP']) else 0.0,
                'Diastolic BP': float(sample_row['Diastolic BP']) if not pd.isna(sample_row['Diastolic BP']) else 0.0,
                'Smoking Status': str(sample_row['Smoking Status']) if not pd.isna(sample_row['Smoking Status']) else 'N',
                'Diabetes Status': str(sample_row['Diabetes Status']) if not pd.isna(sample_row['Diabetes Status']) else 'N',
                'Physical Activity Level': str(sample_row['Physical Activity Level']) if not pd.isna(sample_row['Physical Activity Level']) else 'Low',
                'Family History of CVD': str(sample_row['Family History of CVD']) if not pd.isna(sample_row['Family History of CVD']) else 'N',
                'Height (cm)': float(sample_row['Height (cm)']) if not pd.isna(sample_row['Height (cm)']) else 0.0,
                'Abdominal Circumference (cm)': float(sample_row['Abdominal Circumference (cm)']) if not pd.isna(sample_row['Abdominal Circumference (cm)']) else 0.0,
                'Total Cholesterol (mg/dL)': float(sample_row['Total Cholesterol (mg/dL)']) if not pd.isna(sample_row['Total Cholesterol (mg/dL)']) else 0.0,
                'HDL (mg/dL)': float(sample_row['HDL (mg/dL)']) if not pd.isna(sample_row['HDL (mg/dL)']) else 0.0,
                'Fasting Blood Sugar (mg/dL)': float(sample_row['Fasting Blood Sugar (mg/dL)']) if not pd.isna(sample_row['Fasting Blood Sugar (mg/dL)']) else 0.0,
                'Estimated LDL (mg/dL)': float(sample_row['Estimated LDL (mg/dL)']) if not pd.isna(sample_row['Estimated LDL (mg/dL)']) else 0.0
            }
            
            df_test = pd.DataFrame([data])
            
            # Handle categorical features
            categorical_features = ['Sex', 'Smoking Status', 'Diabetes Status', 'Physical Activity Level', 'Family History of CVD']
            
            for feature in categorical_features:
                if feature in df_test.columns:
                    if feature == 'Sex':
                        df_test[feature] = df_test[feature].map({'M': 1, 'F': 0}).fillna(0)
                    elif feature == 'Smoking Status':
                        df_test[feature] = df_test[feature].map({'Y': 1, 'N': 0}).fillna(0)
                    elif feature == 'Diabetes Status':
                        df_test[feature] = df_test[feature].map({'Y': 1, 'N': 0}).fillna(0)
                    elif feature == 'Physical Activity Level':
                        df_test[feature] = df_test[feature].map({'High': 2, 'Moderate': 1, 'Low': 0}).fillna(0)
                    elif feature == 'Family History of CVD':
                        df_test[feature] = df_test[feature].map({'Y': 1, 'N': 0}).fillna(0)
            
            # Calculate derived features
            df_test['BMI'] = df_test['Weight (kg)'] / (df_test['Height (m)'] ** 2)
            df_test['Waist-to-Height Ratio'] = df_test['Abdominal Circumference (cm)'] / df_test['Height (cm)']
            
            # Handle NaN values
            df_test['BMI'] = df_test['BMI'].fillna(0.0)
            df_test['Waist-to-Height Ratio'] = df_test['Waist-to-Height Ratio'].fillna(0.0)
            
            # Prepare clinical features
            clinical_features = fusion_params['clinical_features']
            clinical_data = []
            for feature in clinical_features:
                if feature in df_test.columns:
                    value = df_test[feature].iloc[0]
                    if pd.isna(value) or np.isnan(value):
                        clinical_data.append(0.0)
                    else:
                        clinical_data.append(value)
                else:
                    clinical_data.append(0.0)
            
            # Prepare lifestyle features
            lifestyle_features = fusion_params['lifestyle_features']
            lifestyle_data = []
            for feature in lifestyle_features:
                if feature in df_test.columns:
                    value = df_test[feature].iloc[0]
                    if pd.isna(value) or np.isnan(value):
                        lifestyle_data.append(0.0)
                    else:
                        lifestyle_data.append(value)
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
            
            predictions.append(risk_level)
            expected.append(sample_row['CVD Risk Level'])
            
            print(f"Sample {i+1:2d}: Expected={sample_row['CVD Risk Level']:12s} Direct={risk_level:12s} Conf={confidence:5.1f}% Probs={probabilities}")
        
        # Analyze direct model predictions
        print(f"\n" + "="*60)
        print("DIRECT MODEL PREDICTION DISTRIBUTION:")
        print("="*60)
        direct_counts = pd.Series(predictions).value_counts()
        print(direct_counts)
        
        print(f"\nExpected distribution:")
        expected_counts = pd.Series(expected).value_counts()
        print(expected_counts)
        
        # Check accuracy by risk level
        print(f"\nAccuracy by risk level:")
        for risk_level in ['LOW', 'INTERMEDIARY', 'HIGH']:
            expected_indices = [i for i, label in enumerate(expected) if label == risk_level]
            if expected_indices:
                correct = sum(1 for i in expected_indices if predictions[i] == risk_level)
                total = len(expected_indices)
                accuracy = correct / total * 100
                print(f"{risk_level}: {correct}/{total} = {accuracy:.1f}%")
        
        return predictions, expected
        
    except Exception as e:
        print(f"Error in direct testing: {e}")
        import traceback
        traceback.print_exc()
        return [], []

if __name__ == "__main__":
    print("=" * 80)
    print("TESTING MODEL DIRECTLY (WITHOUT FLASK APP)")
    print("=" * 80)
    
    predictions, expected = test_model_directly()
    
    if predictions:
        print(f"\nDirect model testing completed!")
    else:
        print("Direct model testing failed!")
