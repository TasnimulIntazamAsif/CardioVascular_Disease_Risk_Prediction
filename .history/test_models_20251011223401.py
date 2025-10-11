#!/usr/bin/env python3
"""
Test different models to find the best one
"""

import pandas as pd
import numpy as np
import joblib

def test_different_models():
    """Test different available models"""
    try:
        # Load dataset
        df = pd.read_csv('CVD Dataset.csv')
        df = df.dropna(subset=['CVD Risk Level'])
        
        print("Testing different models...")
        print(f"Dataset shape: {df.shape}")
        
        # Test models
        models_to_test = [
            'rf_weighted_fusion_model.joblib',
            'rf_fusion_best_model.joblib', 
            'rf_model.joblib',
            'xgboost_fusion_model.joblib'
        ]
        
        for model_name in models_to_test:
            try:
                print(f"\n" + "="*60)
                print(f"Testing {model_name}")
                print("="*60)
                
                model = joblib.load(f'models/{model_name}')
                print(f"Model type: {type(model)}")
                
                # Test with first 10 samples
                predictions = []
                expected = []
                
                for i in range(min(10, len(df))):
                    sample_row = df.iloc[i]
                    
                    # Simple preprocessing for testing
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
                    
                    # Simple feature preparation
                    features = [
                        data['Age'],
                        1 if data['Sex'] == 'M' else 0,
                        data['Weight (kg)'],
                        data['Height (m)'],
                        data['Systolic BP'],
                        data['Diastolic BP'],
                        1 if data['Smoking Status'] == 'Y' else 0,
                        1 if data['Diabetes Status'] == 'Y' else 0,
                        {'High': 2, 'Moderate': 1, 'Low': 0}.get(data['Physical Activity Level'], 0),
                        1 if data['Family History of CVD'] == 'Y' else 0,
                        data['Height (cm)'],
                        data['Abdominal Circumference (cm)'],
                        data['Total Cholesterol (mg/dL)'],
                        data['HDL (mg/dL)'],
                        data['Fasting Blood Sugar (mg/dL)'],
                        data['Estimated LDL (mg/dL)']
                    ]
                    
                    # Calculate BMI and Waist-to-Height Ratio
                    bmi = data['Weight (kg)'] / (data['Height (m)'] ** 2) if data['Height (m)'] > 0 else 0
                    waist_height_ratio = data['Abdominal Circumference (cm)'] / data['Height (cm)'] if data['Height (cm)'] > 0 else 0
                    
                    features.extend([bmi, waist_height_ratio])
                    
                    # Make prediction
                    X = np.array(features).reshape(1, -1)
                    
                    try:
                        prediction = model.predict(X)[0]
                        probabilities = model.predict_proba(X)[0]
                        
                        # Map to risk levels
                        risk_mapping = {0: 'LOW', 1: 'INTERMEDIARY', 2: 'HIGH'}
                        risk_level = risk_mapping.get(prediction, 'UNKNOWN')
                        confidence = max(probabilities) * 100
                        
                        predictions.append(risk_level)
                        expected.append(sample_row['CVD Risk Level'])
                        
                        print(f"Sample {i+1:2d}: Expected={sample_row['CVD Risk Level']:12s} Predicted={risk_level:12s} Conf={confidence:5.1f}%")
                        
                    except Exception as e:
                        print(f"Sample {i+1:2d}: Prediction error - {str(e)[:50]}...")
                        predictions.append('ERROR')
                        expected.append(sample_row['CVD Risk Level'])
                
                # Analyze results
                if predictions:
                    print(f"\nPrediction distribution:")
                    pred_counts = pd.Series(predictions).value_counts()
                    print(pred_counts)
                    
                    print(f"\nExpected distribution:")
                    exp_counts = pd.Series(expected).value_counts()
                    print(exp_counts)
                    
                    # Calculate accuracy
                    correct = sum(1 for p, e in zip(predictions, expected) if p == e)
                    total = len(predictions)
                    accuracy = correct / total * 100 if total > 0 else 0
                    print(f"\nAccuracy: {correct}/{total} = {accuracy:.1f}%")
                
            except Exception as e:
                print(f"Error testing {model_name}: {e}")
        
    except Exception as e:
        print(f"Error in testing: {e}")

if __name__ == "__main__":
    print("=" * 80)
    print("TESTING DIFFERENT MODELS")
    print("=" * 80)
    
    test_different_models()
